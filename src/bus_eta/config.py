"""Project-wide paths and constants."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Borough: BUS_ETA_BOROUGH=brooklyn runs the whole pipeline on Brooklyn under data/brooklyn/.
# Manhattan (the default) keeps the original paths, so nothing changes for it.
BOROUGHS = {
    "manhattan": {
        "data": "data",
        "route_regex": r"^M",                    # all M routes, incl. SBS "+" variants
        "feed": "gtfs_m",
        "bbox": (40.68, -74.03, 40.89, -73.90),  # (S, W, N, E) with margin
        "dot_boros": ["MAN"],                    # NYC DOT bus-lane dataset codes
        "weather_point": (40.7829, -73.9654),    # Central Park
        "events_csv": "major_manhattan_events.csv",
    },
    "brooklyn": {
        "data": "data/brooklyn",
        # B + digit: Brooklyn local and SBS; excludes Bronx (BX..., upper case in RT) and
        # express (BM...). B100 / B103 are MTA Bus Company routes, absent from gtfs_b, so
        # they drop out at the trip join (5% of Brooklyn pings).
        "route_regex": r"^B[0-9]",
        "feed": "gtfs_b",
        "bbox": (40.55, -74.06, 40.76, -73.76),  # Brooklyn plus the B routes into Queens / JFK
        "dot_boros": ["BK", "QNS"],
        "weather_point": (40.6602, -73.9690),    # Prospect Park
        "events_csv": "major_brooklyn_events.csv",  # only the West Indian Day Parade (Labor Day)
        # archived static versions on parquet.gtfsrt.io (pick -> feed digest), see fetch_static
        "schedule_b64": "aHR0cDovL3dlYi5tdGEuaW5mby9kZXZlbG9wZXJzL2RhdGEvbnljdC9idXMvZ29vZ2xlX3RyYW5zaXRfYnJvb2tseW4uemlw",
        # The pick names follow Manhattan's. The Feb 26 version has the same 58,321 trips as A6 with a
        # shorter calendar (to Mar 28), so it is left out (trip_ids must be unique across picks).
        "schedule_versions": {
            "A6": "v1:449017fc7d90309d9d965b7e0fb0be19b3fa04ede856ee038cc193d2879a9828",  # 2026-03-28: Jan 3 - Apr 11
            "B6": "v1:c1d7dea939706fe34d6891ffae240675c4abbf453be52b3f7945190ebec10d6f",  # 2026-04-09: Apr 11 - Jun 27
            "C6": "v1:dd338b4a1f9843baa7a6b10211b8d1d91f28dbe304a2a9f9636ed8f47039bf47",  # 2026-06-24: Jun 27 - Sep 5
            "D6": "v1:111d223dadbb9e58e71b6e376f00dd788ddd857b9480f99c73a74c5d62cdb2dd",  # 2026-09-03: Sep 5 - Jan 2
        },
    },
}
BOROUGH = os.environ.get("BUS_ETA_BOROUGH", "manhattan")
assert BOROUGH in BOROUGHS, f"BUS_ETA_BOROUGH must be one of {list(BOROUGHS)}"
B = BOROUGHS[BOROUGH]

DATA = ROOT / B["data"]
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
    "service_alerts": "aHR0cHM6Ly9ndGZzcnQucHJvZC5vYmFueWMuY29tL2FsZXJ0cw",
}

ROUTE_REGEX = B["route_regex"]
FEED = B["feed"]
STATIC_GTFS_URL = f"https://rrgtfsfeeds.s3.amazonaws.com/{FEED}.zip"

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
