"""Turn per-day wide sample files into a long (issue x horizon) modelling table.

Issues are subsampled deterministically by hashing (trip_id, start_date, k), so the same
issue is always in or out regardless of how the files are read. Only features that are
stable across schedule picks are used (route_id/stop_id, never trip_id/pattern_id), since
the test period starts a new pick.
"""
import polars as pl

from bus_eta.config import H, INTERIM, PROCESSED, SPLITS, TIMEZONE
from bus_eta.parallel import day_range

SAMPLES = PROCESSED / "samples"
CONTEXT = INTERIM / "context"

PER_H = ["y", "sched", "run_sched", "dist", "rt", "rt_fresh", "mta"]
ISSUE_COLS = [
    "trip_id", "start_date", "service_date", "route_id", "direction_id", "stop_id", "k", "t_issue",
    "delay_dep", "delay_arr_k1", "delay_arr_k2", "delay_arr_k3", "pace_ratio_3", "headway_s",
    "issue_lag", "n_stops", "trip_progress",
]
# Test issues scored (deterministic hash subsample); every model must use the same value.
EVAL_FRAC = 0.25
RANDOM_VAL_PPM = 7_000  # 0.7% of issues form the validation set of the random-split pitfall

CATEGORICAL = ["route_id", "stop_id"]
NUMERIC_BASE = [
    "h", "direction_id", "k", "n_stops", "trip_progress",
    "dist", "run_sched", "sched", "rt", "rt_fresh",
    "delay_dep", "delay_arr_k1", "delay_arr_k2", "delay_arr_k3", "pace_ratio_3", "headway_s", "issue_lag",
    "rt_per_sched", "sched_minus_rt",
    "tod_hour", "dow", "is_weekend", "is_holiday", "n_major_events",
    "temp_c", "precip_mm", "snow_cm", "wind_kmh",
]
NUMERIC_MTA = ["mta", "mta_minus_rt"]
NUMERIC_SCHOOL = ["is_school_day"]
# physical attributes of the path from stop k to stop k+h (segment_attrs.py)
NUMERIC_ATTRS = ["path_signals", "path_signals_per_km", "path_lane_share", "path_lane_active_share"]
GRAPH = INTERIM / "graph"


def sample_files(split: str) -> list[str]:
    # random_train / random_val (the "random split" pitfall) draw from every day, train..test
    start, end = (SPLITS["train"][0], SPLITS["test"][1]) if split.startswith("random") else SPLITS[split]
    return [str(SAMPLES / f"{d.isoformat()}.parquet") for d in day_range(start, end)
            if (SAMPLES / f"{d.isoformat()}.parquet").exists()]


def to_long(wide: pl.LazyFrame) -> pl.LazyFrame:
    parts = [
        wide.select(ISSUE_COLS + [pl.lit(h, dtype=pl.Int8).alias("h")]
                    + [pl.col(f"{v}_h{h}").alias(v) for v in PER_H])
        for h in range(1, H + 1)
    ]
    return pl.concat(parts, how="vertical_relaxed")


def stop_segments() -> pl.LazyFrame:
    """(trip_id, stop_idx) -> corridor segment node used to reach that stop."""
    st = pl.scan_parquet(INTERIM / "static" / "stop_times.parquet").select("trip_id", "stop_idx", "stop_id")
    nodes = pl.scan_parquet(GRAPH / "nodes.parquet").select("node", "from_stop", "to_stop")
    return (st.sort("trip_id", "stop_idx")
            .with_columns(pl.col("stop_id").shift(1).over("trip_id").alias("from_stop"))
            .join(nodes, left_on=["from_stop", "stop_id"], right_on=["from_stop", "to_stop"])
            .select("trip_id", "stop_idx", pl.col("node").cast(pl.Int64)))


def add_path_attrs(long: pl.LazyFrame) -> pl.LazyFrame:
    """Signals and bus lanes summed over the segments from stop k to stop k+h. Bus-lane
    share "active" counts only lanes in force at the issue hour (part-time lanes)."""
    import numpy as np  # local: only needed for the lane profile

    attrs = pl.scan_parquet(GRAPH / "node_attrs.parquet").select(
        pl.col("node").cast(pl.Int64), "seg_len_m", "n_signals",
        (pl.col("lane_ded_frac") + pl.col("lane_shared_frac")).clip(upper_bound=1.0).alias("lane_frac"))
    active = np.load(GRAPH / "node_lane_active.npy")
    act = pl.DataFrame({"node": np.repeat(np.arange(active.shape[0]), 168),
                        "how": np.tile(np.arange(168), active.shape[0]).astype(np.int16),
                        "lane_active": active.reshape(-1)}).lazy()
    local = pl.from_epoch(pl.col("t_issue").cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC") \
        .dt.convert_time_zone(TIMEZONE)
    issue = ["trip_id", "start_date", "k"]
    cum = lambda e: e.fill_null(0).cum_sum().over(issue, order_by="h")  # noqa: E731
    path_len = cum(pl.col("seg_len_m")).clip(lower_bound=1.0)
    return (
        long.with_columns((pl.col("k") + pl.col("h")).cast(pl.Int32).alias("stop_idx"),
                          ((local.dt.weekday() - 1) * 24 + local.dt.hour()).cast(pl.Int16).alias("how"))
        .join(stop_segments().with_columns(pl.col("stop_idx").cast(pl.Int32)), on=["trip_id", "stop_idx"], how="left")
        .join(attrs, on="node", how="left")
        .join(act, on=["node", "how"], how="left")
        .with_columns(
            cum(pl.col("n_signals")).alias("path_signals"),
            (cum(pl.col("n_signals")) / path_len * 1000).alias("path_signals_per_km"),
            (cum(pl.col("lane_frac") * pl.col("seg_len_m")) / path_len).alias("path_lane_share"),
            (cum(pl.col("lane_active") * pl.col("seg_len_m")) / path_len).alias("path_lane_active_share"),
        )
        .drop("stop_idx", "how", "node", "seg_len_m", "n_signals", "lane_frac", "lane_active")
    )


def load(split: str, issue_frac: float = 1.0, seed: int = 0, attrs: bool = False) -> pl.DataFrame:
    """Long table for a split: one row per (issue, horizon) with an observed target."""
    wide = pl.scan_parquet(sample_files(split), include_file_paths="_file")
    # two vehicles occasionally report the same trip_id; such issues are ambiguous, drop them
    wide = wide.filter(pl.len().over("trip_id", "start_date", "k") == 1)
    key = pl.concat_str([pl.col("trip_id"), pl.col("start_date"), pl.col("k").cast(pl.Utf8)])
    if split.startswith("random"):
        # Pitfall experiment: issues from all months shuffled into train/val, minus the scored test
        # issues (same EVAL_FRAC hash as load("test")). Neighbouring issues of the same trip share
        # most of their target stops, so this split leaks; it is here only to measure how much.
        # load("test") selects by daily file, not service_date: the Sep 1 file holds late trips of
        # Aug 31, so the test range must be matched on the file date too.
        start, end = SPLITS["test"]
        file_date = pl.col("_file").str.extract(r"(\d{4}-\d{2}-\d{2})\.parquet$").str.to_date()
        scored = file_date.is_between(start, end) & ((key.hash(0) % 1_000_000) < int(EVAL_FRAC * 1_000_000))
        is_val = (key.hash(12345) % 1_000_000) < RANDOM_VAL_PPM
        wide = wide.filter(~scored & (is_val if split == "random_val" else ~is_val))
        seed += 101  # subsample with a hash independent of the scored-issue hash
    if issue_frac < 1.0:
        wide = wide.filter((key.hash(seed) % 1_000_000) < int(issue_frac * 1_000_000))
    long = to_long(wide)
    if attrs:  # needs every horizon, observed or not, for the cumulative sums
        long = add_path_attrs(long)
    long = long.filter(pl.col("y").is_not_null())

    weather = pl.read_parquet(CONTEXT / "weather_hourly.parquet").lazy()
    calendar = pl.read_parquet(CONTEXT / "calendar.parquet").lazy().select(
        "service_date", "is_holiday", "n_major_events", pl.col("is_school_day").cast(pl.Int8))
    local = pl.from_epoch(pl.col("t_issue").cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC") \
        .dt.convert_time_zone(TIMEZONE)
    long = (
        long.with_columns(local.alias("_local"))
        .with_columns(
            (pl.col("_local").dt.hour() + pl.col("_local").dt.minute() / 60).alias("tod_hour"),
            pl.col("_local").dt.weekday().alias("dow"),
            (pl.col("_local").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
            pl.col("_local").dt.replace_time_zone(None).dt.truncate("1h").alias("hour_local"),
        )
        .join(weather, on="hour_local", how="left")
        .join(calendar, on="service_date", how="left")
        .with_columns(
            pl.col("is_holiday").cast(pl.Int8),
            (pl.col("rt") / pl.col("run_sched").clip(lower_bound=30)).alias("rt_per_sched"),
            (pl.col("sched") - pl.col("rt")).alias("sched_minus_rt"),
            (pl.col("mta") - pl.col("rt")).alias("mta_minus_rt"),
        )
        .drop("_local", "hour_local")
    )
    return long.collect(engine="streaming")


def feature_columns(use_mta: bool, school: bool = False, attrs: bool = False) -> list[str]:
    return (CATEGORICAL + NUMERIC_BASE + (NUMERIC_MTA if use_mta else []) + (NUMERIC_SCHOOL if school else [])
            + (NUMERIC_ATTRS if attrs else []))
