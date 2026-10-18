"""Download daily GTFS-RT archives (vehicle positions, trip updates, service alerts) for the borough's routes
(config.BOROUGH; Manhattan by default).

Resumable: days whose output files already exist are skipped, and files are written
atomically (tmp + rename), so an interrupted run never leaves a truncated parquet.

    python -m bus_eta.fetch_rt --start 2026-03-01 --end 2026-10-12

Trip updates are stored "change-only": for each (trip, start_date, stop_sequence) we keep
a snapshot row only when its predicted arrival_time changed, plus the final snapshot in
which the stop still appeared. The prediction in force at any feed time t is therefore the
latest kept row with feed_timestamp <= t, and the stop's disappearance time is preserved.

Service alerts are route-level free text (MTA leaves cause/effect empty). They are reduced
to one row per (alert, route) with the first/last feed time it was seen that day.
"""
import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from bus_eta.config import RAW_RT, ROUTE_REGEX, RT_BASE_URL, RT_FEED_B64, RT_FIRST_DATE

VP_COLS = [
    "feed_timestamp", "timestamp", "vehicle_id", "trip_id", "route_id", "direction_id",
    "start_date", "latitude", "longitude", "bearing", "stop_id", "occupancy_status",
]
TU_COLS = [
    "feed_timestamp", "trip_id", "route_id", "direction_id", "start_date",
    "schedule_relationship", "vehicle_id", "stop_sequence", "stop_id", "arrival_time",
]
TU_KEY = ["trip_id", "start_date", "stop_sequence"]
SA_KEY = ["entity_id", "route_id"]


def feed_url(kind: str, d: date) -> str:
    return f"{RT_BASE_URL}/{kind}/date={d.isoformat()}/base64url={RT_FEED_B64[kind]}/data.parquet"


def scan_borough(kind: str, d: date, cols: list[str]) -> pl.DataFrame:
    return (
        pl.scan_parquet(feed_url(kind, d))
        .filter(pl.col("route_id").str.contains(ROUTE_REGEX))
        .select(cols)
        .collect(engine="streaming")
    )


def fetch_vehicle_positions(d: date) -> pl.DataFrame:
    vp = scan_borough("vehicle_positions", d, VP_COLS)
    # The same GPS ping is repeated across feed snapshots until the bus reports again.
    return vp.unique(subset=["vehicle_id", "timestamp"], keep="first").sort(["vehicle_id", "timestamp"])


def fetch_trip_updates(d: date) -> pl.DataFrame:
    tu = scan_borough("trip_updates", d, TU_COLS).sort(TU_KEY + ["feed_timestamp"])
    changed = (pl.col("arrival_time") != pl.col("arrival_time").shift(1).over(TU_KEY)).fill_null(True)
    last_seen = pl.col("feed_timestamp") == pl.col("feed_timestamp").max().over(TU_KEY)
    return tu.filter(changed | last_seen)


def fetch_service_alerts(d: date) -> pl.DataFrame:
    sa = scan_borough("service_alerts", d, SA_KEY + [
        "feed_timestamp", "active_period_start", "active_period_end", "stop_id", "header_text", "description_text"])
    return sa.group_by(SA_KEY).agg(
        pl.col("feed_timestamp").min().alias("first_seen"), pl.col("feed_timestamp").max().alias("last_seen"),
        pl.col("active_period_start").min(), pl.col("active_period_end").max(),
        pl.col("stop_id").drop_nulls().unique().alias("stop_ids"),
        pl.col("header_text").first(), pl.col("description_text").first(),
    ).sort(SA_KEY)


def write_atomic(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.rename(path)


def fetch_day(d: date, retries: int = 3) -> bool:
    jobs = {"vehicle_positions": fetch_vehicle_positions, "trip_updates": fetch_trip_updates,
            "service_alerts": fetch_service_alerts}
    ok = True
    for kind, fn in jobs.items():
        out = RAW_RT / kind / f"{d.isoformat()}.parquet"
        if out.exists():
            continue
        for attempt in range(1, retries + 1):
            try:
                t0 = time.time()
                df = fn(d)
                write_atomic(df, out)
                print(f"{d} {kind}: {df.height:,} rows in {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as e:  # network/object-store errors: retry, then report
                print(f"{d} {kind}: attempt {attempt} failed: {str(e)[:200]}", flush=True)
                time.sleep(10 * attempt)
        else:
            ok = False
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, default=RT_FIRST_DATE)
    p.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    args = p.parse_args()

    failed = []
    d = args.start
    while d <= args.end:
        if not fetch_day(d):
            failed.append(d.isoformat())
        d += timedelta(days=1)
    print(f"done; failed days: {failed or 'none'}")


if __name__ == "__main__":
    main()
