"""Build static tables for every schedule pick in data/raw/gtfs_static/<PICK>/<feed>.zip
(config.FEED: gtfs_m for Manhattan, gtfs_b for Brooklyn).

Outputs (data/interim/static/), every table keyed by `pick` because shape_id/stop
geometry can differ between picks:
  trips.parquet         pick, trip_id, route_id, direction_id, service_id, shape_id, block_id, pattern_id
  stop_times.parquet    pick, trip_id, stop_sequence, stop_id, sched_arr_s, sched_dep_s, dist_m
  shapes.parquet        pick, shape_id, x, y, cum_m           (points in shape order)
  stops.parquet         pick, stop_id, stop_name, lat, lon, x, y
  service_dates.parquet pick, service_id, service_date

dist_m is the stop's distance along its trip's shape, computed once per stop pattern.

    python -m bus_eta.static_gtfs
"""
import io
import zipfile
from datetime import date, timedelta

import numpy as np
import polars as pl

from bus_eta.config import FEED, INTERIM, RAW_STATIC, ROUTE_REGEX
from bus_eta.geo import cumulative_distance, project_stops, to_xy

OUT = INTERIM / "static"
MAX_STOP_OFFSET_M = 150.0  # a stop further than this from its shape is reported


def read_txt(zf: zipfile.ZipFile, name: str) -> pl.DataFrame:
    return pl.read_csv(io.BytesIO(zf.read(name)), infer_schema=False)


def hms_to_seconds(col: str) -> pl.Expr:
    p = pl.col(col).str.strip_chars().str.split(":")
    return (p.list.get(0).cast(pl.Int32) * 3600 + p.list.get(1).cast(pl.Int32) * 60
            + p.list.get(2).cast(pl.Int32))


def resolve_service_dates(calendar: pl.DataFrame, calendar_dates: pl.DataFrame) -> pl.DataFrame:
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    rows = []
    for r in calendar.iter_rows(named=True):
        d = date.fromisoformat(f"{r['start_date'][:4]}-{r['start_date'][4:6]}-{r['start_date'][6:]}")
        end = date.fromisoformat(f"{r['end_date'][:4]}-{r['end_date'][4:6]}-{r['end_date'][6:]}")
        while d <= end:
            if r[days[d.weekday()]] == "1":
                rows.append((r["service_id"], d))
            d += timedelta(days=1)
    base = pl.DataFrame(rows, schema=["service_id", "service_date"], orient="row")
    exc = calendar_dates.with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d").alias("service_date"))
    added = exc.filter(pl.col("exception_type") == "1").select(base.columns)
    removed = exc.filter(pl.col("exception_type") == "2").select(base.columns)
    return pl.concat([base, added]).unique().join(removed, on=base.columns, how="anti")


def build_pick(pick: str) -> dict[str, pl.DataFrame]:
    zf = zipfile.ZipFile(RAW_STATIC / pick / f"{FEED}.zip")

    # Only the borough's routes, as in the RT archive (gtfs_b also carries 9 Queens and 4
    # express routes run from Brooklyn depots; gtfs_m is all M routes, so a no-op there).
    trips = read_txt(zf, "trips.txt").filter(pl.col("route_id").str.contains(ROUTE_REGEX)).select(
        "trip_id", "route_id", pl.col("direction_id").cast(pl.Int8), "service_id", "shape_id", "block_id")

    stops = read_txt(zf, "stops.txt").select(
        "stop_id", "stop_name",
        pl.col("stop_lat").str.strip_chars().cast(pl.Float64).alias("lat"),
        pl.col("stop_lon").str.strip_chars().cast(pl.Float64).alias("lon"))
    x, y = to_xy(stops["lat"].to_numpy(), stops["lon"].to_numpy())
    stops = stops.with_columns(pl.Series("x", x), pl.Series("y", y))

    shapes = (
        read_txt(zf, "shapes.txt")
        .select("shape_id", pl.col("shape_pt_sequence").cast(pl.Int32),
                pl.col("shape_pt_lat").cast(pl.Float64), pl.col("shape_pt_lon").cast(pl.Float64))
        .sort("shape_id", "shape_pt_sequence")
    )
    x, y = to_xy(shapes["shape_pt_lat"].to_numpy(), shapes["shape_pt_lon"].to_numpy())
    shapes = shapes.with_columns(pl.Series("x", x), pl.Series("y", y))
    shape_geom = {}
    for (sid,), g in shapes.group_by("shape_id"):
        xs, ys = g["x"].to_numpy(), g["y"].to_numpy()
        shape_geom[sid] = (xs, ys, cumulative_distance(xs, ys))
    shapes_out = pl.concat([
        pl.DataFrame({"shape_id": sid, "x": xs, "y": ys, "cum_m": cum})
        for sid, (xs, ys, cum) in shape_geom.items()
    ])

    stop_times = (
        read_txt(zf, "stop_times.txt")
        .select("trip_id", "stop_id", pl.col("stop_sequence").cast(pl.Int32),
                hms_to_seconds("arrival_time").alias("sched_arr_s"),
                hms_to_seconds("departure_time").alias("sched_dep_s"))
        .join(trips.select("trip_id"), on="trip_id", how="semi")
        .sort("trip_id", "stop_sequence")
    )

    # One pattern per (shape, ordered stop list): projection is computed once per pattern.
    sig = stop_times.group_by("trip_id", maintain_order=True).agg(pl.col("stop_id").str.join(",").alias("stop_list"))
    trips = trips.join(sig, on="trip_id", how="inner")
    patterns = (
        trips.select("shape_id", "stop_list").unique().sort("shape_id", "stop_list")
        .with_row_index("pattern_id").with_columns(pl.format("{}_{}", pl.lit(pick), "pattern_id").alias("pattern_id"))
    )
    trips = trips.join(patterns, on=["shape_id", "stop_list"]).drop("stop_list")

    stop_xy = dict(zip(stops["stop_id"].to_list(), zip(stops["x"].to_list(), stops["y"].to_list())))
    pat_rows, worst = [], []
    for r in patterns.iter_rows(named=True):
        ids = r["stop_list"].split(",")
        xs, ys, cum = shape_geom[r["shape_id"]]
        sx = np.array([stop_xy[s][0] for s in ids])
        sy = np.array([stop_xy[s][1] for s in ids])
        dist, lateral = project_stops(sx, sy, xs, ys, cum)
        worst.append(float(np.max(lateral)))
        pat_rows.append(pl.DataFrame({
            "pattern_id": r["pattern_id"], "stop_idx": np.arange(len(ids), dtype=np.int32),
            "dist_m": dist, "stop_offset_m": lateral}))
    pattern_dist = pl.concat(pat_rows)
    worst = np.array(worst)
    print(f"[{pick}] {trips.height:,} trips, {len(patterns):,} patterns, {len(shape_geom)} shapes; "
          f"patterns with a stop > {MAX_STOP_OFFSET_M:.0f} m off-shape: {(worst > MAX_STOP_OFFSET_M).sum()}")

    stop_times = (
        stop_times.join(trips.select("trip_id", "pattern_id"), on="trip_id")
        .with_columns((pl.col("stop_sequence").rank("ordinal").over("trip_id") - 1).cast(pl.Int32).alias("stop_idx"))
        .join(pattern_dist, on=["pattern_id", "stop_idx"])
        .select("trip_id", "stop_sequence", "stop_idx", "stop_id", "sched_arr_s", "sched_dep_s", "dist_m", "stop_offset_m")
    )

    service_dates = resolve_service_dates(read_txt(zf, "calendar.txt"), read_txt(zf, "calendar_dates.txt"))

    tag = pl.lit(pick).alias("pick")
    return {
        "trips": trips.select(tag, pl.all()),
        "stop_times": stop_times.select(tag, pl.all()),
        "shapes": shapes_out.select(tag, pl.all()),
        "stops": stops.select(tag, pl.all()),
        "service_dates": service_dates.select(tag, pl.all()),
    }


def main() -> None:
    picks = sorted(p.name for p in RAW_STATIC.iterdir() if (p / f"{FEED}.zip").exists())
    built = [build_pick(p) for p in picks]
    # Downstream joins key on trip_id alone, so a trip_id may appear in only one pick.
    ids = pl.concat([b["trips"].select("trip_id") for b in built])
    assert ids["trip_id"].is_unique().all(), f"trip_ids repeated across picks {picks}"
    OUT.mkdir(parents=True, exist_ok=True)
    for name in built[0]:
        df = pl.concat([b[name] for b in built])
        df.write_parquet(OUT / f"{name}.parquet")
        print(f"{name}: {df.shape}")


if __name__ == "__main__":
    main()
