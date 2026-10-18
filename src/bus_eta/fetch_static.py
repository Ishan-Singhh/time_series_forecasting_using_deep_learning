"""Static GTFS for the borough (config.BOROUGH) into data/raw/gtfs_static/<PICK>/<feed>.zip.

MTA only serves the current feed, so each run keeps a dated copy; trip_ids in the
real-time archive must be resolved against the version that was live on that day.

    python -m bus_eta.fetch_static               # snapshot of the current feed
    python -m bus_eta.fetch_static --archived    # every version in config (parquet.gtfsrt.io)

--archived rebuilds a GTFS zip from the archive's per-table parquet files (all columns are
stored as the original text, so the CSVs round-trip). Each version goes to a pick folder
named by its retrieval date; static_gtfs drops trip_ids repeated in a later version.
"""
import argparse
import io
import zipfile
from datetime import date

import polars as pl
import requests

from bus_eta.config import B, FEED, RAW_STATIC, RT_BASE_URL, STATIC_GTFS_URL

TABLES = ["agency", "routes", "trips", "stop_times", "stops", "shapes", "calendar", "calendar_dates"]


def snapshot() -> None:
    out = RAW_STATIC / date.today().isoformat() / f"{FEED}.zip"
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


def archived_zip(b64: str, digest: str, out) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for t in TABLES:
            df = pl.read_parquet(f"{RT_BASE_URL}/schedules/base64url={b64}/_feed_digest={digest}/{t}.parquet")
            assert all(dt == pl.String for dt in df.dtypes), f"{t}: non-text columns {df.schema}"
            # blank fields are "" in the archive but empty (null) in MTA's CSVs
            df = df.with_columns(pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c) for c in df.columns)
            buf = io.BytesIO()
            df.write_csv(buf)
            zf.writestr(f"{t}.txt", buf.getvalue())
            print(f"  {t}: {df.height:,} rows", flush=True)
    tmp.rename(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archived", action="store_true", help="fetch the archived versions listed in config")
    args = p.parse_args()
    if not args.archived:
        return snapshot()
    for pick, digest in B["schedule_versions"].items():
        out = RAW_STATIC / pick / f"{FEED}.zip"
        if out.exists():
            print(f"already have {out}")
            continue
        print(f"{pick} {digest[:16]}", flush=True)
        archived_zip(B["schedule_b64"], digest, out)


if __name__ == "__main__":
    main()
