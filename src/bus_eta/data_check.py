"""Data-quality report for the current borough (config.BOROUGH) over a range of days.

Run it on a new borough before training and compare with Manhattan on the same days:
every number should be of the same order, and the hard checks (marked FAIL) must be 0.

  RT        pings, share of pings whose trip_id is in the static picks
  arrivals  trips, stop events, labelled arrivals, long interpolation gaps,
            FAIL: trip keys in more than one day file, duplicate stop rows,
                  arrivals that go back in time along a trip, service date != file day
  samples   issues, targets per issue, MTA / fresh real-time coverage, baseline errors,
            FAIL: negative targets (arrival before the issue), targets decreasing with h

    BUS_ETA_BOROUGH=brooklyn python -m bus_eta.data_check --start 2026-03-02 --end 2026-03-06
"""
import argparse
from datetime import date

import polars as pl

from bus_eta.config import BOROUGH, H, INTERIM, PROCESSED, RAW_RT
from bus_eta.parallel import day_range


def check(days: list[date]) -> dict:
    r, fails = {"borough": BOROUGH, "days": len(days)}, {}
    static_ids = pl.read_parquet(INTERIM / "static" / "trips.parquet", columns=["trip_id"])["trip_id"]

    vp = pl.concat([pl.read_parquet(RAW_RT / "vehicle_positions" / f"{d}.parquet", columns=["trip_id", "route_id"])
                    for d in days])
    r["pings/day"] = vp.height / len(days)
    r["routes"] = vp["route_id"].n_unique()
    r["pings in static %"] = 100 * vp["trip_id"].is_in(static_ids.implode()).mean()

    arr = pl.concat([pl.read_parquet(INTERIM / "arrivals" / f"{d}.parquet").with_columns(pl.lit(d).alias("file_day"))
                     for d in days])
    key = ["trip_id", "start_date"]
    r["trips/day"] = arr.select(key).unique().height / len(days)
    r["stop events/day"] = arr.height / len(days)
    r["labelled arrivals %"] = 100 * arr["t_arr"].is_not_null().mean()
    r["arr gap > 60 s %"] = 100 * (arr["arr_gap_s"] > 60).mean()
    r["median stops/trip"] = arr.group_by(key).len()["len"].median()
    fails["trip keys in >1 file"] = arr.group_by(key).agg(pl.col("file_day").n_unique()).filter(pl.col("file_day") > 1).height
    fails["duplicate stop rows"] = arr.height - arr.select(key + ["stop_idx"]).unique().height
    fails["service_date != file day"] = arr.filter(pl.col("service_date") != pl.col("file_day")).height
    back = (arr.filter(pl.col("t_arr").is_not_null()).sort(key + ["stop_idx"])
            .with_columns((pl.col("t_arr") - pl.col("t_arr").shift(1).over(key)).alias("dt")))
    fails["arrivals back in time"] = back.filter(pl.col("dt") < -1e-6).height

    s = pl.concat([pl.read_parquet(PROCESSED / "samples" / f"{d}.parquet") for d in days], how="diagonal")
    y = [f"y_h{h}" for h in range(1, H + 1)]
    r["issues/day"] = s.height / len(days)
    r["targets/issue"] = s.select(pl.sum_horizontal(pl.col(c).is_not_null() for c in y)).to_series().mean()
    r["median y_h1 s"] = s["y_h1"].median()
    r["median y_h20 s"] = s["y_h20"].median()
    r["MTA available h1 %"] = 100 * s["mta_h1"].is_not_null().mean()
    r["rt_fresh h5 mean"] = s["rt_fresh_h5"].mean()
    r["median delay_dep s"] = s["delay_dep"].median()
    long = pl.concat([s.select(pl.col(f"y_h{h}").alias("y"), pl.col(f"{m}_h{h}").alias("p"), pl.lit(m).alias("m"))
                      for h in range(1, H + 1) for m in ("sched", "rt", "mta")]).drop_nulls()
    for (m,), g in long.group_by("m"):
        r[f"MAE {m} s"] = (g["p"] - g["y"]).abs().mean()
    fails["negative targets"] = sum(s[c].lt(0).sum() for c in y)
    fails["targets decreasing with h"] = sum(
        s.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null() & (pl.col(b) < pl.col(a) - 1e-6)).height
        for a, b in zip(y[:-1], y[1:]))
    r["FAIL"] = {k: v for k, v in fails.items()}
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    args = p.parse_args()
    r = check(list(day_range(args.start, args.end)))
    for k, v in r.items():
        print(f"{k:28s} {v:,.2f}" if isinstance(v, float) else f"{k:28s} {v}")
    bad = {k: v for k, v in r["FAIL"].items() if v}
    print("CHECKS PASSED" if not bad else f"CHECKS FAILED: {bad}")


if __name__ == "__main__":
    main()
