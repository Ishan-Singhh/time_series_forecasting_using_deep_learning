"""Multi-step ETA samples: one row per (bus departs stop k, observed at t_issue).

t_issue is the timestamp of the GPS ping that revealed the departure (dep_seen), so every
input below is something a real-time system had at that moment. For each horizon
h = 1..H the row carries:

  y_h          observed t_arr(k+h) - t_issue                (target; null if unreliable)
  sched_h      scheduled arrival at k+h - t_issue            (encodes current lateness)
  run_sched_h  scheduled running time k -> k+h
  dist_h       distance along the route k -> k+h (m)
  rt_h         "real-time profile" ETA: most recent observed running/dwell times of ANY bus
               on each downstream segment (shared stop pairs across routes), revealed
               before t_issue and at most RT_MAX_AGE_S old; schedule fills the gaps
  rt_fresh_h   fraction of the h segments that had fresh real-time information
  mta_h        MTA Bus Time's prediction in force at t_issue - t_issue

plus bus-state features (current/previous delays, own recent pace, headway to the
previous bus of the same route at this stop).

Output: data/processed/samples/<YYYY-MM-DD>.parquet

    python -m bus_eta.samples --start 2026-03-01 --end 2026-10-11
"""
import argparse
import os
from datetime import date, timedelta

import polars as pl

from bus_eta.config import H, INTERIM, PROCESSED, RAW_RT, n_cpus
from bus_eta.parallel import day_range, run_days, write_atomic

ARRIVALS = INTERIM / "arrivals"
STATIC = INTERIM / "static"
OUT = PROCESSED / "samples"


MAX_LABEL_GAP_S = 90.0    # target dropped if its arrival was interpolated over a longer ping gap
RT_MAX_AGE_S = 1800       # real-time segment observations older than this are ignored
MTA_MAX_STALE_S = 60      # MTA prediction counts only if the stop was still in the feed
TRIP = ["trip_id", "start_date"]
LONG_VALUES = ["y", "sched", "run_sched", "dist", "rt", "rt_fresh", "mta"]


def read_arrivals(d: date) -> pl.DataFrame | None:
    p = ARRIVALS / f"{d.isoformat()}.parquet"
    return pl.read_parquet(p) if p.exists() else None


def traversals(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Segment running times and stop dwell times, each stamped with when it became known."""
    e = events.sort(TRIP + ["stop_idx"])
    nxt = e.select(TRIP + [(pl.col("stop_idx") - 1).alias("stop_idx"), pl.col("stop_id").alias("to_stop"),
                           "t_arr", "arr_seen", "arr_gap_s"])
    runs = (
        e.select(TRIP + ["stop_idx", pl.col("stop_id").alias("from_stop"), "t_dep", "dep_gap_s"])
        .join(nxt, on=TRIP + ["stop_idx"])
        .filter((pl.col("arr_gap_s") <= MAX_LABEL_GAP_S) & (pl.col("dep_gap_s") <= MAX_LABEL_GAP_S))
        .select("from_stop", "to_stop", (pl.col("t_arr") - pl.col("t_dep")).alias("rt_run"),
                pl.col("arr_seen").alias("known_at"))
        .drop_nulls().sort("known_at")
    )
    dwells = (
        e.filter((pl.col("stop_idx") > 0) & (pl.col("arr_gap_s") <= MAX_LABEL_GAP_S)
                 & (pl.col("dep_gap_s") <= MAX_LABEL_GAP_S))
        .select(pl.col("stop_id").alias("dwell_stop"), (pl.col("t_dep") - pl.col("t_arr")).alias("rt_dwell"),
                pl.col("dep_seen").alias("known_at"))
        .drop_nulls().sort("known_at")
    )
    return runs, dwells


def mta_predictions(d: date, trips: pl.DataFrame) -> pl.DataFrame | None:
    # service day d spans the UTC archive files of d and d+1 (see arrivals.py)
    paths = [p for p in (RAW_RT / "trip_updates" / f"{x.isoformat()}.parquet" for x in (d, d + timedelta(days=1)))
             if p.exists()]
    if not paths:
        return None
    tu = (
        pl.read_parquet(paths, columns=["feed_timestamp", "trip_id", "start_date", "stop_sequence", "arrival_time"])
        .join(trips, on=TRIP, how="semi")
        .with_columns(pl.col("feed_timestamp").cast(pl.Float64), pl.col("stop_sequence").cast(pl.Int32))
    )
    key = TRIP + ["stop_sequence"]
    return (
        tu.with_columns(pl.col("feed_timestamp").max().over(key).alias("mta_last_seen"))
        .rename({"feed_timestamp": "mta_feed_ts", "arrival_time": "mta_arrival"})
        .sort("mta_feed_ts")
    )


def build_day(d: date, static_st: pl.DataFrame) -> pl.DataFrame | None:
    today = read_arrivals(d)
    if today is None:
        return None
    # Other buses' observations also come from the neighbouring service days: late trips of d-1
    # and early trips of d+1 overlap d's night hours. Every use is an as-of join on the time an
    # observation was revealed, so including d+1 cannot leak.
    days = (read_arrivals(d - timedelta(days=1)), today, read_arrivals(d + timedelta(days=1)))
    history = pl.concat([x for x in days if x is not None])
    runs, dwells = traversals(history)

    issues = (
        today.filter(pl.col("dep_seen").is_not_null() & (pl.col("dep_gap_s") <= MAX_LABEL_GAP_S))
        .select(TRIP + ["pick", "service_date", "route_id", "direction_id", "pattern_id", "vehicle_id",
                        pl.col("stop_idx").alias("k"), "stop_id", "dist_m",
                        pl.col("dep_seen").alias("t_issue"), "t_dep", "sched_dep"])
        .with_row_index("issue_id")
    )
    if issues.height == 0:
        return None
    midnight = (pl.col("sched_dep") - pl.col("sched_dep_s"))  # epoch of the service day's 00:00
    issues = issues.join(static_st.select("trip_id", pl.col("stop_idx").alias("k"), "sched_dep_s"),
                         on=["trip_id", "k"]).with_columns(midnight.alias("midnight")).drop("sched_dep_s")

    # ---------------- long grid: one row per (issue, h) ----------------
    st_j = static_st.select("trip_id", pl.col("stop_idx").alias("j"), pl.col("stop_id").alias("stop_j"),
                            "stop_sequence", pl.col("sched_arr_s").alias("sched_arr_j_s"), pl.col("dist_m").alias("dist_j"))
    st_prev = static_st.select("trip_id", (pl.col("stop_idx") + 1).alias("j"), pl.col("stop_id").alias("stop_prev"),
                               pl.col("sched_dep_s").alias("sched_dep_prev_s"))
    long = (
        issues.select("issue_id", *TRIP, "k", "t_issue", "t_dep", "dist_m", "midnight")
        .join(pl.DataFrame({"h": list(range(1, H + 1))}, schema={"h": pl.Int32}), how="cross")
        .with_columns((pl.col("k") + pl.col("h")).cast(pl.Int32).alias("j"))
        .join(st_j, on=["trip_id", "j"])
        .join(st_prev, on=["trip_id", "j"])
        .with_columns(
            (pl.col("midnight") + pl.col("sched_arr_j_s") - pl.col("t_issue")).alias("sched"),
            (pl.col("sched_arr_j_s") - pl.col("sched_dep_prev_s")).alias("seg_sched"),
            (pl.col("dist_j") - pl.col("dist_m")).alias("dist"),
        )
    )

    # Targets: own trip's observed arrival at k+h, if reliable and not already known at t_issue.
    own = today.select(TRIP + [pl.col("stop_idx").alias("j"), "t_arr", "arr_gap_s", "arr_seen"])
    long = long.join(own, on=TRIP + ["j"], how="left").with_columns(
        pl.when((pl.col("arr_gap_s") <= MAX_LABEL_GAP_S) & (pl.col("arr_seen") > pl.col("t_issue")))
        .then(pl.col("t_arr") - pl.col("t_issue")).alias("y")
    ).drop("t_arr", "arr_gap_s", "arr_seen")

    # Real-time: latest revealed running time on segment (stop_prev -> stop_j), and dwell at stop_prev.
    long = (
        long.sort("t_issue")
        .join_asof(runs, left_on="t_issue", right_on="known_at", by_left=["stop_prev", "stop_j"],
                   by_right=["from_stop", "to_stop"], strategy="backward", tolerance=RT_MAX_AGE_S, check_sortedness=False)
        .drop("known_at")
        .join_asof(dwells, left_on="t_issue", right_on="known_at", by_left="stop_prev", by_right="dwell_stop",
                   strategy="backward", tolerance=RT_MAX_AGE_S, check_sortedness=False)
        .drop("known_at")
        .sort("issue_id", "h")
        .with_columns(
            pl.when(pl.col("h") > 1).then(pl.col("rt_dwell").fill_null(0.0)).otherwise(0.0).alias("dwell_used"),
            pl.col("rt_run").is_not_null().cast(pl.Float32).alias("_fresh"),
        )
        .with_columns(
            pl.col("seg_sched").cum_sum().over("issue_id").alias("run_sched"),
            ((pl.col("rt_run").fill_null(pl.col("seg_sched")) + pl.col("dwell_used")).cum_sum().over("issue_id")
             - (pl.col("t_issue") - pl.col("t_dep"))).alias("rt"),
            (pl.col("_fresh").cum_sum().over("issue_id") / pl.col("h")).alias("rt_fresh"),
        )
    )

    # MTA prediction in force at t_issue (stop must still have been in the feed).
    mta = mta_predictions(d, issues.select(TRIP).unique())
    if mta is None:
        long = long.with_columns(pl.lit(None, dtype=pl.Float64).alias("mta"))
    else:
        long = (
            long.with_columns(pl.col("stop_sequence").cast(pl.Int32)).sort("t_issue")
            .join_asof(mta, left_on="t_issue", right_on="mta_feed_ts", by=TRIP + ["stop_sequence"],
                       strategy="backward", check_sortedness=False)
            .with_columns(
                pl.when(pl.col("mta_last_seen") >= pl.col("t_issue") - MTA_MAX_STALE_S)
                .then(pl.col("mta_arrival") - pl.col("t_issue")).cast(pl.Float64).alias("mta"))
        )

    wide = long.select(["issue_id", "h"] + LONG_VALUES).pivot(on="h", index="issue_id", values=LONG_VALUES)
    wide = wide.rename({f"{v}_{h}": f"{v}_h{h}" for v in LONG_VALUES for h in range(1, H + 1)}, strict=False)

    # ---------------- bus-state features ----------------
    def own_at(offset: int, cols: dict) -> pl.DataFrame:
        return today.select(TRIP + [(pl.col("stop_idx") + offset).alias("k")] + [e.alias(n) for n, e in cols.items()])

    delay_cols = {f"delay_arr_k{o}": (pl.col("t_arr") - pl.col("sched_arr")) for o in (1, 2, 3)}
    feats = issues.with_columns(
        (pl.col("t_dep") - pl.col("sched_dep")).alias("delay_dep"),
        (pl.col("t_issue") - pl.col("t_dep")).alias("issue_lag"),
    )
    for o in (1, 2, 3):
        feats = feats.join(own_at(o, {f"delay_arr_k{o}": delay_cols[f"delay_arr_k{o}"]}), on=TRIP + ["k"], how="left")
    feats = (
        feats.join(own_at(3, {"_t_dep_k3": pl.col("t_dep"), "_sched_dep_k3": pl.col("sched_dep")}),
                   on=TRIP + ["k"], how="left")
        .with_columns(((pl.col("t_dep") - pl.col("_t_dep_k3"))
                       / (pl.col("sched_dep") - pl.col("_sched_dep_k3")).clip(lower_bound=30)).alias("pace_ratio_3"))
        .drop("_t_dep_k3", "_sched_dep_k3")
    )
    # Headway: previous departure of another bus of the same route/direction from this stop.
    deps = (history.filter(pl.col("dep_seen").is_not_null())
            .select("route_id", "direction_id", "stop_id", pl.col("t_dep").alias("prev_bus_dep"),
                    pl.col("dep_seen").alias("_seen")).sort("_seen"))
    feats = (
        feats.sort("t_issue")
        .join_asof(deps, left_on="t_issue", right_on="_seen", by=["route_id", "direction_id", "stop_id"],
                   strategy="backward", allow_exact_matches=False, check_sortedness=False)
        .with_columns((pl.col("t_dep") - pl.col("prev_bus_dep")).alias("headway_s"))
        .drop("_seen", "prev_bus_dep", "midnight")
    )
    n_stops = static_st.group_by("trip_id").agg(pl.len().alias("n_stops"))
    feats = feats.join(n_stops, on="trip_id").with_columns((pl.col("k") / pl.col("n_stops")).alias("trip_progress"))

    out = feats.join(wide, on="issue_id").sort("t_issue")
    return out.filter(pl.any_horizontal(pl.col(f"y_h{h}").is_not_null() for h in range(1, H + 1)
                                        if f"y_h{h}" in out.columns))


_ST = None


def _init():
    global _ST
    _ST = pl.read_parquet(STATIC / "stop_times.parquet",
                          columns=["trip_id", "stop_idx", "stop_sequence", "stop_id", "sched_arr_s",
                                   "sched_dep_s", "dist_m"])


# set by --force in main(); an env var so the spawned day workers see it too
FORCE = os.environ.get("BUS_ETA_FORCE") == "1"


def _run(d: date) -> str:
    out = OUT / f"{d.isoformat()}.parquet"
    if out.exists() and not FORCE:
        return f"{d} exists"
    df = build_day(d, _ST)
    if df is None:
        return f"{d} no data"
    write_atomic(df, out)
    return f"{d} {df.height:,} samples"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    # Each day builds a ~5M-row long grid, so use half the cores as workers (2 threads each).
    p.add_argument("--workers", type=int, default=max(1, n_cpus() // 2))
    p.add_argument("--force", action="store_true", help="rebuild days whose output already exists")
    args = p.parse_args()
    if args.force:
        os.environ["BUS_ETA_FORCE"] = "1"
    run_days(_run, day_range(args.start, args.end), args.workers, initializer=_init)


if __name__ == "__main__":
    main()
