"""Project-wide paths and constants."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_RT = DATA / "raw" / "rt"
RAW_STATIC = DATA / "raw" / "gtfs_static"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

# parquet.gtfsrt.io archives the MTA Bus Time feeds from 2026-03-01 onward.
RT_FIRST_DATE = date(2026, 3, 1)
RT_BASE_URL = "http://parquet.gtfsrt.io"
RT_FEED_B64 = {
    "trip_updates": "aHR0cHM6Ly9ndGZzcnQucHJvZC5vYmFueWMuY29tL3RyaXBVcGRhdGVz",
    "vehicle_positions": "aHR0cHM6Ly9ndGZzcnQucHJvZC5vYmFueWMuY29tL3ZlaGljbGVQb3NpdGlvbnM",
}

# All Manhattan routes (M*, including SBS "+" variants).
ROUTE_PREFIX = "M"

STATIC_GTFS_URL = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_m.zip"

TIMEZONE = "America/New_York"


def n_cpus() -> int:
    """Cores allocated to this job: PBS sets NCPUS; os.cpu_count() would report the whole node."""
    return int(os.environ.get("NCPUS") or os.cpu_count() or 1)

# Chronological split. Test never informs training, early stopping or tuning.
SPLITS = {
    "train": (date(2026, 3, 1), date(2026, 7, 31)),
    "val": (date(2026, 8, 1), date(2026, 8, 31)),
    "test": (date(2026, 9, 1), date(2026, 10, 11)),
}
H = 20  # prediction horizons: next H stops
