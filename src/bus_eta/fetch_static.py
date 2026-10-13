"""Snapshot MTA's current Manhattan static GTFS into data/raw/gtfs_static/<YYYY-MM-DD>/.

MTA only serves the current feed, so each run keeps a dated copy; trip_ids in the
real-time archive must be resolved against the version that was live on that day.

    python -m bus_eta.fetch_static
"""
from datetime import date

import requests

from bus_eta.config import RAW_STATIC, STATIC_GTFS_URL


def main() -> None:
    out = RAW_STATIC / date.today().isoformat() / "gtfs_m.zip"
    if out.exists():
        print(f"already have {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(STATIC_GTFS_URL, timeout=120)
    resp.raise_for_status()
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(resp.content)
    tmp.rename(out)
    print(f"saved {out} ({len(resp.content) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
