"""Exogenous context: hourly weather (Open-Meteo, Central Park) and a daily calendar.

    python -m bus_eta.context --start 2026-03-01 --end 2026-10-11    # needs internet: login node

Outputs data/interim/context/{weather_hourly,calendar}.parquet
"""
import argparse
from datetime import date

import polars as pl
import requests

from bus_eta.config import INTERIM, ROOT, TIMEZONE
from bus_eta.parallel import write_atomic

OUT = INTERIM / "context"
CENTRAL_PARK = (40.7829, -73.9654)

# US federal holidays in the study period (observed dates).
FEDERAL_HOLIDAYS = [
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 10, 12),  # Columbus / Indigenous Peoples' Day
]


def fetch_weather(start: date, end: date) -> pl.DataFrame:
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": CENTRAL_PARK[0], "longitude": CENTRAL_PARK[1],
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": "temperature_2m,precipitation,rain,snowfall,windspeed_10m,weathercode",
            "timezone": TIMEZONE,
        },
        timeout=60,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return pl.DataFrame({
        "hour_local": h["time"], "temp_c": h["temperature_2m"], "precip_mm": h["precipitation"],
        "rain_mm": h["rain"], "snow_cm": h["snowfall"], "wind_kmh": h["windspeed_10m"],
        "weathercode": h["weathercode"],
    }).with_columns(pl.col("hour_local").str.strptime(pl.Datetime("us"), "%Y-%m-%dT%H:%M"))


def school_days(days: pl.Series) -> pl.Series:
    """NYC public-school in-person days from nyc_school_calendar.csv (transcribed from the
    NYCPS 2025-26 and 2026-27 calendars): a weekday inside a school year and not a closure,
    recess or remote-only day. Early-dismissal days count as school days."""
    cal = pl.read_csv(ROOT / "nyc_school_calendar.csv", try_parse_dates=True)
    d = pl.DataFrame({"service_date": days})
    in_year = pl.lit(False)
    for s, e in cal.filter(pl.col("kind") == "year").select("start", "end").rows():
        in_year = in_year | pl.col("service_date").is_between(s, e)
    closed = pl.lit(False)
    for s, e in cal.filter(pl.col("kind") == "closed").select("start", "end").rows():
        closed = closed | pl.col("service_date").is_between(s, e)
    return d.select((in_year & ~closed & (pl.col("service_date").dt.weekday() <= 5)).alias("is_school_day")).to_series()


def build_calendar(start: date, end: date) -> pl.DataFrame:
    """Daily calendar: school days (official NYCPS calendars), summer break, federal
    holidays and the number of major Manhattan events."""
    days = pl.DataFrame({"service_date": pl.date_range(start, end, "1d", eager=True)})
    events = (
        pl.read_csv(ROOT / "major_manhattan_events.csv")
        .select(pl.col("event_date").str.strptime(pl.Date, "%Y-%m-%d").alias("service_date"))
        .group_by("service_date").len().rename({"len": "n_major_events"})
    )
    return (
        days.with_columns(school_days(days["service_date"]))
        .with_columns(  # between the last day of one school year and the first of the next
            pl.col("service_date").is_between(date(2026, 6, 27), date(2026, 9, 9)).alias("is_summer_break"))
        .join(events, on="service_date", how="left")
        .with_columns(
            pl.col("service_date").is_in(FEDERAL_HOLIDAYS).alias("is_holiday"),
            pl.col("n_major_events").fill_null(0),
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--calendar-only", action="store_true", help="rebuild the calendar, keep weather")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.calendar_only:
        w = fetch_weather(args.start, args.end)
        write_atomic(w, OUT / "weather_hourly.parquet")
        print(f"weather {w.shape}, null temp {w['temp_c'].null_count()}")
    c = build_calendar(args.start, args.end)
    write_atomic(c, OUT / "calendar.parquet")  # running jobs may be reading it
    print(f"calendar {c.shape}, school days {c['is_school_day'].sum()}")


if __name__ == "__main__":
    main()
