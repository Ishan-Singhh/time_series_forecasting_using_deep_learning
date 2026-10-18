"""Manhattan regression: the borough-switch code must reproduce the paper's Manhattan data.

Rebuilds static tables, graph, arrivals (2 days) and samples (1 day) into a scratch folder and
compares them with data/interim and data/processed, which the paper's models used.

    python jobs/checks/regress_manhattan.py <scratch dir>     # repo root, CPU node
"""
import sys
from datetime import date
from pathlib import Path
import polars as pl
import bus_eta.config as cfg
assert cfg.BOROUGH == "manhattan"
TMP = Path(sys.argv[1]); TMP.mkdir(parents=True, exist_ok=True)

def same(name, a: pl.DataFrame, b: pl.DataFrame):
    """Identical rows: sort by a unique key when the table has one (a sort by every column
    misorders rows whose leading columns are null), else by every column."""
    keys = {"arrivals": ["trip_id", "start_date", "stop_idx"], "samples": ["issue_id"], "static/trips": ["pick", "trip_id"],
            "static/stop_times": ["pick", "trip_id", "stop_idx"], "graph/nodes": ["node"]}
    k = next((v for p, v in keys.items() if name.startswith(p)), sorted(a.columns))
    ok = (a.schema == b.schema or set(a.columns) == set(b.columns)) and a.height == b.height and \
        a.select(sorted(a.columns)).sort(k).equals(b.select(sorted(a.columns)).sort(k))
    print(f"{name}: {'IDENTICAL' if ok else 'DIFFERENT'} {a.shape} vs {b.shape}", flush=True)
    return ok

import bus_eta.static_gtfs as sg
sg.OUT = TMP / "static"; sg.main()
oks = [same(f"static/{t}", pl.read_parquet(TMP / "static" / f"{t}.parquet"), pl.read_parquet(cfg.INTERIM / "static" / f"{t}.parquet"))
       for t in ["trips", "stop_times", "shapes", "stops", "service_dates"]]

import bus_eta.graph as gr
gr.OUT = TMP / "graph"; gr.main()
oks += [same(f"graph/{t}", pl.read_parquet(TMP / "graph" / f"{t}.parquet"), pl.read_parquet(cfg.INTERIM / "graph" / f"{t}.parquet"))
        for t in ["nodes", "edges"]]

import bus_eta.arrivals as ar
static = ar.load_static()
for d in [date(2026, 3, 8), date(2026, 9, 16)]:          # DST day and a test-period weekday
    oks.append(same(f"arrivals/{d}", ar.build_day(d, static), pl.read_parquet(cfg.INTERIM / "arrivals" / f"{d}.parquet")))

import bus_eta.samples as sm
sm._init()
d = date(2026, 9, 16)
oks.append(same(f"samples/{d}", sm.build_day(d, sm._ST), pl.read_parquet(cfg.PROCESSED / "samples" / f"{d}.parquet")))
print("MANHATTAN REGRESSION", "PASSED" if all(oks) else "FAILED")
