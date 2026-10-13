"""Where do the models win or lose? Test MAE broken down by operating condition.

Groups (each scored on the common support of the compared models, all horizons):
  period     time of day: night 0-6, AM peak 6-10, midday 10-16, PM peak 16-20, evening 20-24
  daytype    weekday / weekend
  rain       hourly precipitation at issue time: dry (0), light (<1 mm), heavy (>=1 mm)
  route_type Select Bus Service ('+' suffix) vs local
  alert      a service alert was live for the route at issue time (feed first..last seen)
  school     before / after the first school day (Sep 10)
  route      every route (separate file)
For each group: rows, days, MAE per model, skill of each model vs mta_cal and the graph
gain (no-graph minus graph model).

    python -m bus_eta.error_analysis
    python -m bus_eta.error_analysis --models mta_cal lgbm_mta seq_mta stgnn_mta
"""
import argparse
from datetime import date

import polars as pl

from bus_eta.config import DATA, INTERIM
from bus_eta.evaluate import RESULTS, load_common

TZ = "America/New_York"
SCHOOL_START = date(2026, 9, 10)
PERIODS = [(0, 6, "night 0-6"), (6, 10, "AM peak 6-10"), (10, 16, "midday 10-16"),
           (16, 20, "PM peak 16-20"), (20, 24, "evening 20-24")]


def alert_windows(days: list[date]) -> pl.DataFrame:
    """(route_id, start, end) in UTC epoch seconds for every alert seen on the test days."""
    parts = []
    for d in days:
        f = DATA / "raw" / "rt" / "service_alerts" / f"{d.isoformat()}.parquet"
        if f.exists():
            parts.append(pl.read_parquet(f, columns=["route_id", "first_seen", "last_seen"]))
    return pl.concat(parts).drop_nulls().unique().with_columns(pl.col("first_seen", "last_seen").cast(pl.Int64))


def add_conditions(df: pl.DataFrame) -> pl.DataFrame:
    hour = pl.col("tod_hour")
    period = pl.lit(PERIODS[-1][2])
    for lo, hi, name in reversed(PERIODS[:-1]):
        period = pl.when(hour.is_between(lo, hi, closed="left")).then(pl.lit(name)).otherwise(period)
    # dow (ISO, 1 = Monday) and tod_hour are of the issue's calendar date, which is the day
    # after service_date for trips running past midnight
    next_day = (pl.col("dow") - pl.col("service_date").dt.weekday()) % 7
    issue_local = (pl.col("service_date").cast(pl.Datetime("us")) + pl.duration(days=next_day)
                   + pl.duration(seconds=(hour * 3600).cast(pl.Int64)))
    df = df.with_columns(
        period.alias("period"),
        pl.when(pl.col("dow") >= 6).then(pl.lit("weekend")).otherwise(pl.lit("weekday")).alias("daytype"),
        pl.when(pl.col("route_id").str.ends_with("+")).then(pl.lit("SBS")).otherwise(pl.lit("local")).alias("route_type"),
        pl.when(pl.col("service_date") >= SCHOOL_START).then(pl.lit("school (Sep 10+)"))
          .otherwise(pl.lit("no school (Sep 1-9)")).alias("school"),
        issue_local.alias("issue_local"),
    ).with_columns(
        pl.col("issue_local").dt.truncate("1h").alias("hour_local"),
        pl.col("issue_local").dt.replace_time_zone(TZ, ambiguous="earliest", non_existent="null")
          .dt.epoch("s").alias("issue_utc"),
    )
    weather = pl.read_parquet(INTERIM / "context" / "weather_hourly.parquet", columns=["hour_local", "precip_mm"])
    df = df.join(weather, on="hour_local", how="left").with_columns(
        pl.when(pl.col("precip_mm") <= 0).then(pl.lit("dry"))
          .when(pl.col("precip_mm") < 1).then(pl.lit("light rain <1mm/h"))
          .otherwise(pl.lit("heavy rain >=1mm/h")).alias("rain"))

    # alert live for the route at issue time: interval join on the distinct (route, issue) pairs
    issues = df.select("route_id", "issue_utc").unique().drop_nulls()
    al = alert_windows(df["service_date"].unique().to_list())
    hit = (issues.join(al, on="route_id").filter(pl.col("issue_utc").is_between(pl.col("first_seen"), pl.col("last_seen")))
           .select("route_id", "issue_utc").unique().with_columns(pl.lit("alert live").alias("alert")))
    return df.join(hit, on=["route_id", "issue_utc"], how="left").with_columns(pl.col("alert").fill_null("no alert"))


def breakdown(df: pl.DataFrame, by: str, models: list[str], graph_pair: tuple[str, str] | None) -> pl.DataFrame:
    err = [(pl.col(m) - pl.col("y")).abs().mean().alias(m) for m in models]
    out = df.group_by(by).agg(pl.len().alias("rows"), pl.col("service_date").n_unique().alias("days"), *err)
    out = out.with_columns([(100 * (1 - pl.col(m) / pl.col("mta_cal"))).alias(f"skill_{m}_%")
                            for m in models if m != "mta_cal"])
    if graph_pair:
        out = out.with_columns((pl.col(graph_pair[0]) - pl.col(graph_pair[1])).alias("graph_gain_s"))
    return out.rename({by: "group"}).with_columns(pl.lit(by).alias("factor")).sort("group")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=["mta_cal", "mta", "lgbm_mta", "seq_mta", "stgnn_mta"])
    p.add_argument("--name", default="error_analysis")
    args = p.parse_args()
    assert "mta_cal" in args.models, "skill is reported vs mta_cal"
    pair = ("seq_mta", "stgnn_mta") if {"seq_mta", "stgnn_mta"} <= set(args.models) else None

    df = add_conditions(load_common(args.models))
    tables = [breakdown(df, f, args.models, pair)
              for f in ("period", "daytype", "rain", "route_type", "alert", "school")]
    res = pl.concat(tables, how="diagonal").select("factor", "group", pl.exclude("factor", "group"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.write_csv(RESULTS / f"{args.name}.csv")
    by_route = breakdown(df, "route_id", args.models, pair).drop("factor").sort("rows", descending=True)
    by_route.write_csv(RESULTS / f"{args.name}_by_route.csv")

    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=250, float_precision=1, fmt_str_lengths=24):
        print(res)
        print(by_route)
    print(f"wrote {RESULTS / args.name}.csv and _by_route.csv")


if __name__ == "__main__":
    main()
