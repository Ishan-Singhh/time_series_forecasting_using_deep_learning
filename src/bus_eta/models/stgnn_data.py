"""Pack samples into dense per-issue arrays for the neural models.

For each split writes data/processed/nn/<split>.npz with, per issue (rows sorted by bin):
  bin        int32   5-min bin of t_issue, relative to network_state first_bin
  route      int32  route index (0 = unknown)
  dow        int8   0..6
  issue_x    float32 (F_issue,)   bus state, time of day, weather, calendar
  tok_x      float32 (H, F_tok)   per-horizon features (dist, run_sched, sched, rt, rt_fresh, mta, has_mta)
  tok_valid  bool    (H,)        stop k+h exists on the trip (an input-side mask, never target-based)
  tok_node   int32   (H,)        segment node crossed to reach stop k+h (-1 = unknown)
  rt         float32 (H,)        real-time profile ETA (prediction = rt + model output)
  y          float32 (H,)        target (NaN = unobserved)
  key        trip_id/start_date/k for writing predictions

Normalisation statistics are fitted on train and stored in nn/stats.json.

    python -m bus_eta.models.stgnn_data --train-frac 0.2 --val-frac 0.1
"""
import argparse
import json

import numpy as np
import polars as pl

from bus_eta.config import H, INTERIM, PROCESSED, TIMEZONE
from bus_eta.features import CONTEXT, EVAL_FRAC, sample_files

OUT = PROCESSED / "nn"
GRAPH = INTERIM / "graph"
STATE_META = PROCESSED / "network_state" / "meta.json"

ISSUE_NUM = ["delay_dep", "delay_arr_k1", "delay_arr_k2", "delay_arr_k3", "pace_ratio_3", "headway_s",
             "issue_lag", "trip_progress", "k", "n_stops", "tod_sin", "tod_cos", "is_holiday",
             "n_major_events", "temp_c", "precip_mm", "snow_cm", "wind_kmh"]
TOK_NUM = ["dist", "run_sched", "sched", "rt", "rt_fresh", "mta"]


def load_wide(split: str, frac: float, seed: int) -> pl.DataFrame:
    wide = pl.scan_parquet(sample_files(split))
    # two vehicles occasionally report the same trip_id; such issues are ambiguous, drop them
    wide = wide.filter(pl.len().over("trip_id", "start_date", "k") == 1)
    if frac < 1.0:
        key = pl.concat_str([pl.col("trip_id"), pl.col("start_date"), pl.col("k").cast(pl.Utf8)])
        wide = wide.filter((key.hash(seed) % 1_000_000) < int(frac * 1_000_000))
    weather = pl.read_parquet(CONTEXT / "weather_hourly.parquet").lazy()
    calendar = pl.read_parquet(CONTEXT / "calendar.parquet").lazy().select("service_date", "is_holiday", "n_major_events")
    local = (pl.from_epoch(pl.col("t_issue").cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC")
             .dt.convert_time_zone(TIMEZONE))
    tod = (local.dt.hour() + local.dt.minute() / 60) * (2 * np.pi / 24)
    return (
        wide.with_columns(tod.sin().alias("tod_sin"), tod.cos().alias("tod_cos"),
                          (local.dt.weekday() - 1).cast(pl.Int8).alias("dow"),
                          local.dt.replace_time_zone(None).dt.truncate("1h").alias("hour_local"))
        .join(weather, on="hour_local", how="left")
        .join(calendar, on="service_date", how="left")
        .with_columns(pl.col("is_holiday").cast(pl.Float32))
        .collect(engine="streaming")
    )


def downstream_nodes(wide: pl.DataFrame) -> np.ndarray:
    """Segment node crossed to reach stop k+h, for h = 1..H (-1 if past trip end / unknown)."""
    st = pl.read_parquet(INTERIM / "static" / "stop_times.parquet", columns=["trip_id", "stop_idx", "stop_id"])
    nodes = pl.read_parquet(GRAPH / "nodes.parquet", columns=["node", "from_stop", "to_stop"])
    seg = (st.sort("trip_id", "stop_idx")
           .with_columns(pl.col("stop_id").shift(1).over("trip_id").alias("from_stop"))
           .join(nodes, left_on=["from_stop", "stop_id"], right_on=["from_stop", "to_stop"], how="left")
           .select("trip_id", "stop_idx", pl.col("node").cast(pl.Int32).fill_null(-1)))
    base = wide.select("trip_id", "k").with_row_index("_r")
    out = np.full((wide.height, H), -1, dtype=np.int32)
    for h in range(1, H + 1):
        j = base.with_columns((pl.col("k") + h).cast(pl.Int32).alias("stop_idx")).join(
            seg, on=["trip_id", "stop_idx"], how="left").sort("_r")
        out[:, h - 1] = j["node"].fill_null(-1).to_numpy()
    return out


def pack(split: str, wide: pl.DataFrame, stats: dict, route_map: dict, first_bin: int, bin_s: int) -> None:
    wide = wide.with_columns(((pl.col("t_issue") // bin_s).cast(pl.Int64) - first_bin).alias("bin")).sort("bin")

    def mat(cols, names):  # standardised, NaN kept for masks
        x = wide.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
        return (x - np.array([stats[n][0] for n in names], dtype=np.float32)) / np.array(
            [stats[n][1] for n in names], dtype=np.float32)

    issue_x = np.nan_to_num(mat(ISSUE_NUM, ISSUE_NUM))
    tok = np.stack([mat([f"{v}_h{h}" for h in range(1, H + 1)], [v] * H) for v in TOK_NUM], axis=-1)
    has_mta = (~np.isnan(tok[..., TOK_NUM.index("mta")]))[..., None].astype(np.float32)
    tok_valid = ~np.isnan(tok[..., TOK_NUM.index("dist")])  # stop k+h exists on this trip
    tok_x = np.concatenate([np.nan_to_num(tok), has_mta], axis=-1)
    rt = wide.select([pl.col(f"rt_h{h}") for h in range(1, H + 1)]).to_numpy().astype(np.float32)
    y = wide.select([pl.col(f"y_h{h}") for h in range(1, H + 1)]).to_numpy().astype(np.float32)
    np.savez(
        OUT / f"{split}.npz",
        bin=wide["bin"].to_numpy().astype(np.int32),
        route=wide["route_id"].replace_strict(route_map, default=0, return_dtype=pl.Int32).to_numpy(),
        dow=wide["dow"].to_numpy(), issue_x=issue_x, tok_x=tok_x, tok_valid=tok_valid,
        tok_node=downstream_nodes(wide),
        rt=np.nan_to_num(rt), y=y,
        trip_id=wide["trip_id"].to_numpy().astype(str), start_date=wide["start_date"].to_numpy().astype(str),
        k=wide["k"].to_numpy(),
    )
    print(f"{split}: {wide.height:,} issues, {np.isfinite(y).sum():,} targets", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-frac", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(STATE_META.read_text())

    train = load_wide("train", args.train_frac, args.seed)
    stats = {}
    for c in ISSUE_NUM:
        s = train[c].cast(pl.Float64)
        stats[c] = (float(s.mean() or 0.0), float(s.std() or 1.0) or 1.0)
    for v in TOK_NUM:
        s = pl.concat([train[f"{v}_h{h}"].cast(pl.Float64) for h in range(1, H + 1)])
        stats[v] = (float(s.mean() or 0.0), float(s.std() or 1.0) or 1.0)
    route_map = {r: i + 1 for i, r in enumerate(train["route_id"].unique().sort().to_list())}
    (OUT / "stats.json").write_text(json.dumps({"stats": stats, "route_map": route_map,
                                                "issue_num": ISSUE_NUM, "tok_num": TOK_NUM + ["has_mta"]}))

    pack("train", train, stats, route_map, meta["first_bin"], meta["bin_s"])
    del train
    pack("val", load_wide("val", args.val_frac, args.seed), stats, route_map, meta["first_bin"], meta["bin_s"])
    pack("test", load_wide("test", EVAL_FRAC, 0), stats, route_map, meta["first_bin"], meta["bin_s"])


if __name__ == "__main__":
    main()
