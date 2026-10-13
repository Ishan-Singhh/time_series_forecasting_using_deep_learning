"""Exogenous context: hourly weather (Open-Meteo, Central Park) and a daily calendar.

    python -m bus_eta.context --start 2026-03-01 --end 2026-10-11    # needs internet: login node

Outputs data/interim/context/{weather_hourly,calendar}.parquet
"""
import argparse
from datetime import date

import polars as pl
import requests

from bus_eta.config import INTERIM, ROOT, TIMEZONE

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


def build_calendar(start: date, end: date) -> pl.DataFrame:
    """Daily calendar. School-day flags come from the hand-built calendar_dataset.parquet
    where it covers the date; outside it they are null until the NYC DOE calendar is added."""
    days = pl.DataFrame({"service_date": pl.date_range(start, end, "1d", eager=True)})
    manual = (
        pl.read_parquet(ROOT / "calendar_dataset.parquet")
        .select(pl.col("service_date").str.strptime(pl.Date, "%Y-%m-%d"), "is_school_day", "is_summer_break")
    )
    events = (
        pl.read_csv(ROOT / "major_manhattan_events.csv")
        .select(pl.col("event_date").str.strptime(pl.Date, "%Y-%m-%d").alias("service_date"))
        .group_by("service_date").len().rename({"len": "n_major_events"})
    )
    return (
        days.join(manual, on="service_date", how="left")
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
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    w = fetch_weather(args.start, args.end)
    w.write_parquet(OUT / "weather_hourly.parquet")
    c = build_calendar(args.start, args.end)
    c.write_parquet(OUT / "calendar.parquet")
    print(f"weather {w.shape}, null temp {w['temp_c'].null_count()}; calendar {c.shape}, "
          f"school-day unknown on {c['is_school_day'].null_count()} days")


if __name__ == "__main__":
    main()
