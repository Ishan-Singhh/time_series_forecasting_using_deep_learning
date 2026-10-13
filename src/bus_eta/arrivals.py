"""Observed stop arrival/departure times from vehicle positions (the ground truth).

For each trip instance (trip_id, start_date) the GPS track of its main vehicle is projected
onto the trip's shape (bus_eta.geo.project_track), made non-decreasing, and the times the
bus crossed each stop are linearly interpolated between the two bracketing pings:

  t_arr  crossing of (stop distance - ARRIVE_BEFORE_M)
  t_dep  crossing of (stop distance + DEPART_AFTER_M)

arr_gap_s / dep_gap_s are the time gaps between the bracketing pings, i.e. how much
interpolation the label involved; downstream code filters on them. arr_seen / dep_seen are
the timestamps of the ping that revealed each crossing: the earliest moment a real-time
system could know about it, which is what causal features and issue times must use.

Output: data/interim/arrivals/<YYYY-MM-DD>.parquet (keyed by RT archive day).

    python -m bus_eta.arrivals --start 2026-03-01 --end 2026-10-11 --workers 40
"""
import argparse
from datetime import date

import numpy as np
import polars as pl
from numba import njit

from bus_eta.config import INTERIM, RAW_RT, TIMEZONE, n_cpus
from bus_eta.geo import project_track, to_xy
from bus_eta.parallel import day_range, run_days, write_atomic

STATIC = INTERIM / "static"
OUT = INTERIM / "arrivals"

ARRIVE_BEFORE_M = 10.0
DEPART_AFTER_M = 15.0
MAX_LATERAL_M = 75.0     # GPS ping further than this from the route is discarded
MAX_SPEED_MPS = 25.0     # ~56 mph: bounds forward jumps between consecutive pings
ANCHOR_BACK_M = 100.0
MIN_PINGS = 5


@njit(cache=True)
def crossing_times(t, d, thresholds):
    """First time a non-decreasing track d(t) reaches each threshold.

    Returns (interpolated crossing time, gap between the bracketing pings, time of the ping
    that revealed the crossing). Thresholds the track never brackets (already past at the
    first ping, or never reached) get nan.
    """
    m = len(thresholds)
    out_t = np.full(m, np.nan)
    out_gap = np.full(m, np.nan)
    out_seen = np.full(m, np.nan)
    j = 0
    for k in range(m):
        thr = thresholds[k]
        while j < len(d) and d[j] < thr:
            j += 1
        if j == len(d):
            break
        if j == 0:
            continue
        d0, d1 = d[j - 1], d[j]
        t0, t1 = t[j - 1], t[j]
        frac = (thr - d0) / (d1 - d0) if d1 > d0 else 1.0
        out_t[k] = t0 + frac * (t1 - t0)
        out_gap[k] = t1 - t0
        out_seen[k] = t1
    return out_t, out_gap, out_seen


def load_static():
    trips = pl.read_parquet(STATIC / "trips.parquet")
    stop_times = pl.read_parquet(STATIC / "stop_times.parquet")
    shapes = pl.read_parquet(STATIC / "shapes.parquet")
    geom = {
        (p, s): (g["x"].to_numpy(), g["y"].to_numpy(), g["cum_m"].to_numpy())
        for (p, s), g in shapes.group_by("pick", "shape_id")
    }
    return trips, stop_times, geom


def build_day(d: date, static) -> pl.DataFrame | None:
    trips, stop_times, geom = static
    vp_path = RAW_RT / "vehicle_positions" / f"{d.isoformat()}.parquet"
    if not vp_path.exists():
        return None

    vp = (
        pl.read_parquet(vp_path, columns=["trip_id", "start_date", "vehicle_id", "timestamp",
                                          "latitude", "longitude", "stop_id"])
        .drop_nulls(["trip_id", "start_date", "latitude", "longitude", "timestamp"])
        .join(trips.select("pick", "trip_id", "shape_id", "pattern_id"), on="trip_id", how="inner")
    )
    # One vehicle per trip instance: the one that reported it most.
    main_vehicle = (
        vp.group_by("trip_id", "start_date", "vehicle_id").len()
        .sort("len", descending=True).unique(["trip_id", "start_date"], keep="first")
        .filter(pl.col("len") >= MIN_PINGS).drop("len")
    )
    vp = vp.join(main_vehicle, on=["trip_id", "start_date", "vehicle_id"], how="inner")

    # Anchor window from the feed's next-stop field: between the previous stop and the next one.
    pattern_stops = stop_times.select("pick", "trip_id", "stop_idx", "stop_id", "dist_m")
    anchors = (
        pattern_stops.sort("pick", "trip_id", "stop_idx")
        .with_columns(pl.col("dist_m").shift(1).over("pick", "trip_id").fill_null(0.0).alias("prev_dist_m"))
        .group_by("pick", "trip_id", "stop_id").agg(pl.col("prev_dist_m").min(), pl.col("dist_m").min())
        .select("pick", "trip_id", "stop_id",
                (pl.col("prev_dist_m") - ANCHOR_BACK_M).alias("a_lo"), (pl.col("dist_m") + 50.0).alias("a_hi"))
    )
    vp = vp.join(anchors, on=["pick", "trip_id", "stop_id"], how="left").sort("trip_id", "start_date", "timestamp")
    x, y = to_xy(vp["latitude"].to_numpy(), vp["longitude"].to_numpy())
    vp = vp.with_columns(pl.Series("x", x), pl.Series("y", y))

    st_by_trip = {k[0]: g for k, g in pattern_stops.join(
        vp.select("trip_id").unique(), on="trip_id").sort("stop_idx").partition_by("trip_id", as_dict=True).items()}

    parts = []
    for (trip_id, start_date), g in vp.group_by("trip_id", "start_date", maintain_order=True):
        pick, shape_id = g["pick"][0], g["shape_id"][0]
        xs, ys, cum = geom[(pick, shape_id)]
        t = g["timestamp"].to_numpy().astype(np.float64)
        dist, _ = project_track(g["x"].to_numpy(), g["y"].to_numpy(), t,
                                g["a_lo"].fill_null(np.nan).to_numpy(), g["a_hi"].fill_null(np.nan).to_numpy(),
                                xs, ys, cum, MAX_LATERAL_M, MAX_SPEED_MPS)
        ok = ~np.isnan(dist)
        if ok.sum() < MIN_PINGS:
            continue
        t_ok, d_ok = t[ok], np.maximum.accumulate(dist[ok])
        stops = st_by_trip[trip_id]
        sd = stops["dist_m"].to_numpy()
        t_arr, arr_gap, arr_seen = crossing_times(t_ok, d_ok, sd - ARRIVE_BEFORE_M)
        t_dep, dep_gap, dep_seen = crossing_times(t_ok, d_ok, sd + DEPART_AFTER_M)
        parts.append(stops.select("stop_idx", "stop_id", "dist_m").with_columns(
            pl.lit(trip_id).alias("trip_id"), pl.lit(start_date).alias("start_date"),
            pl.Series("t_arr", t_arr), pl.Series("arr_gap_s", arr_gap), pl.Series("arr_seen", arr_seen),
            pl.Series("t_dep", t_dep), pl.Series("dep_gap_s", dep_gap), pl.Series("dep_seen", dep_seen)))
    if not parts:
        return None

    arr = pl.concat(parts).filter(pl.col("t_arr").is_not_null() | pl.col("t_dep").is_not_null())
    arr = arr.with_columns(pl.col(c).fill_nan(None) for c in ["t_arr", "arr_gap_s", "arr_seen", "t_dep", "dep_gap_s", "dep_seen"])
    # Attach trip metadata and the schedule (as epoch seconds) for convenience downstream.
    arr = (
        arr.join(trips.select("pick", "trip_id", "route_id", "direction_id", "pattern_id", "block_id"), on="trip_id")
        .join(stop_times.select("trip_id", "stop_idx", "stop_sequence", "sched_arr_s", "sched_dep_s"),
              on=["trip_id", "stop_idx"])
        .join(main_vehicle, on=["trip_id", "start_date"])
        .with_columns(pl.col("start_date").str.strptime(pl.Date, "%Y%m%d").alias("service_date"))
        .with_columns(
            pl.col("service_date").cast(pl.Datetime("us")).dt.replace_time_zone(TIMEZONE)
            .dt.epoch("s").alias("_midnight"))
        .with_columns((pl.col("_midnight") + pl.col("sched_arr_s")).alias("sched_arr"),
                      (pl.col("_midnight") + pl.col("sched_dep_s")).alias("sched_dep"))
        .drop("_midnight", "sched_arr_s", "sched_dep_s")
    )
    return arr.select(
        "pick", "trip_id", "start_date", "service_date", "route_id", "direction_id", "pattern_id", "block_id",
        "vehicle_id", "stop_idx", "stop_sequence", "stop_id", "dist_m",
        "sched_arr", "t_arr", "arr_gap_s", "arr_seen", "sched_dep", "t_dep", "dep_gap_s", "dep_seen",
    ).sort("trip_id", "start_date", "stop_idx")


_STATIC = None


def _init():
    global _STATIC
    _STATIC = load_static()


def _run(d: date) -> str:
    out = OUT / f"{d.isoformat()}.parquet"
    if out.exists():
        return f"{d} exists"
    df = build_day(d, _STATIC)
    if df is None:
        return f"{d} no data"
    write_atomic(df, out)
    return f"{d} {df.height:,} stop events, {df.select('trip_id', 'start_date').n_unique():,} trips"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--workers", type=int, default=n_cpus())
    args = p.parse_args()
    run_days(_run, day_range(args.start, args.end), args.workers, initializer=_init)


if __name__ == "__main__":
    main()
