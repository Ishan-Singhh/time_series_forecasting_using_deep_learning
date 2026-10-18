"""Physical attributes of each corridor-graph segment: traffic signals and bus lanes.

Sources (fetched on the login node, cached under data/raw/):
  OpenStreetMap highway=traffic_signals nodes (Overpass API)   -> osm/traffic_signals.json
  NYC DOT "Bus Lanes - Local Streets" (ycrg-ses3), the borough -> nyc/bus_lanes.json

Per segment node (straight line between its two stops):
  n_signals          signalised intersections within SIGNAL_RADIUS_M of the segment; OSM
                     nodes closer than SIGNAL_MERGE_M are one intersection (dual carriageways)
  signals_per_km
  lane_ded_frac      share of the segment alongside a dedicated bus lane (any hours)
  lane_shared_frac   ... alongside a shared/curbside bus lane
  seg_len_m          straight stop-to-stop length
and, because most Manhattan bus lanes are part-time, a weekly profile
  lane_active[node, hour_of_week]  share of the segment with a bus lane in force
(hour_of_week = weekday * 24 + hour, Monday = 0, local time). Lane direction is not
matched (only parallel heading), a limitation on two-way streets.

    python -m bus_eta.segment_attrs --fetch     # login node (internet)
    python -m bus_eta.segment_attrs
"""
import argparse
import json
import re

import numpy as np
import polars as pl
import requests
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from bus_eta.config import B, DATA, INTERIM
from bus_eta.geo import to_xy

RAW_SIGNALS = DATA / "raw" / "osm" / "traffic_signals.json"
RAW_LANES = DATA / "raw" / "nyc" / "bus_lanes.json"
GRAPH = INTERIM / "graph"
BBOX = B["bbox"]  # the borough with margin (S, W, N, E)
SIGNAL_RADIUS_M = 15.0
SIGNAL_MERGE_M = 25.0
LANE_RADIUS_M = 12.0
LANE_MIN_COS = 0.85     # parallel within ~32 degrees
STEP_M = 5.0
ATTRS = ["seg_len_m", "n_signals", "signals_per_km", "lane_ded_frac", "lane_shared_frac"]


def fetch() -> None:
    q = f'[out:json][timeout:200];node["highway"="traffic_signals"]({",".join(map(str, BBOX))});out;'
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q},
                      headers={"User-Agent": "bus-eta-research/0.1 (academic)"}, timeout=300)
    r.raise_for_status()
    RAW_SIGNALS.parent.mkdir(parents=True, exist_ok=True)
    RAW_SIGNALS.write_text(r.text)
    r = requests.get("https://data.cityofnewyork.us/resource/ycrg-ses3.json",
                     params={"$where": "boro in(" + ",".join(f"'{b}'" for b in B["dot_boros"]) + ")", "$limit": 50000}, timeout=300)
    r.raise_for_status()
    RAW_LANES.parent.mkdir(parents=True, exist_ok=True)
    RAW_LANES.write_text(r.text)
    print(f"signals {len(json.loads(RAW_SIGNALS.read_text())['elements'])}, lanes {len(r.json())}")


def _hour(h: str, ampm: str) -> int:
    h = int(h) % 12
    return h + (12 if ampm.upper() == "PM" else 0)


def weekly_mask(hours: str | None, days: str | None) -> np.ndarray:
    """168-hour mask of when a lane is in force; unparseable text counts as always."""
    m = np.zeros((7, 24), dtype=bool)
    d = (days or "").lower()
    weekdays = range(7) if ("7 day" in d or not d) else range(6) if "sat" in d else range(5)
    text = (hours or "").upper()
    spans = re.findall(r"(\d{1,2})\s*(AM|PM)\s*-\s*(\d{1,2})\s*(AM|PM)", text)
    if "24" in text or not spans:
        spans_h = [(0, 24)]
    else:
        spans_h = []
        for h0, a0, h1, a1 in spans:
            s, e = _hour(h0, a0), _hour(h1, a1)
            spans_h.append((s, e if e > s else 24))  # "2PM-12PM" = until midnight
    for wd in weekdays:
        for s, e in spans_h:
            m[wd, s:e] = True
    return m.reshape(-1)


def merged_signals() -> np.ndarray:
    el = json.loads(RAW_SIGNALS.read_text())["elements"]
    x, y = to_xy(np.array([e["lat"] for e in el]), np.array([e["lon"] for e in el]))
    pts = np.stack([x, y], 1)
    pairs = cKDTree(pts).query_pairs(SIGNAL_MERGE_M, output_type="ndarray")
    n = len(pts)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, label = connected_components(adj, directed=False)
    return np.stack([np.bincount(label, pts[:, k]) / np.bincount(label) for k in range(2)], 1)


def lane_points():
    """Points every STEP_M along each lane polyline, with unit heading, type and schedule id."""
    lanes = json.loads(RAW_LANES.read_text())
    sched_ids, scheds = {}, []
    xs, ys, hx, hy, ded, sid = [], [], [], [], [], []
    for ln in lanes:
        key = (ln.get("hours"), ln.get("days"))
        if key not in sched_ids:
            sched_ids[key] = len(scheds)
            scheds.append(weekly_mask(*key))
        dedicated = ln.get("lane_type1") == "Bus Lane"
        for line in ln["the_geom"]["coordinates"]:
            lon, lat = np.array(line).T
            px, py = to_xy(lat, lon)
            for i in range(len(px) - 1):
                dx, dy = px[i + 1] - px[i], py[i + 1] - py[i]
                L = np.hypot(dx, dy)
                if L < 0.5:
                    continue
                t = np.arange(0, L, STEP_M) / L
                xs.append(px[i] + t * dx)
                ys.append(py[i] + t * dy)
                hx.append(np.full(len(t), dx / L))
                hy.append(np.full(len(t), dy / L))
                ded.append(np.full(len(t), dedicated))
                sid.append(np.full(len(t), sched_ids[key]))
    cat = np.concatenate
    return (np.stack([cat(xs), cat(ys)], 1), np.stack([cat(hx), cat(hy)], 1), cat(ded), cat(sid),
            np.stack(scheds))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true", help="download the raw sources (needs internet)")
    args = p.parse_args()
    if args.fetch:
        fetch()

    nodes = pl.read_parquet(GRAPH / "nodes.parquet", columns=["node", "x0", "y0", "x1", "y1"]).sort("node")
    a = nodes.select("x0", "y0").to_numpy()
    b = nodes.select("x1", "y1").to_numpy()
    seg = b - a
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    unit = seg / np.maximum(seg_len, 1e-6)[:, None]
    n_nodes = len(nodes)

    # signals: project each merged intersection on the nearby segments; t in [0, 1) so an
    # intersection at a stop is counted once, by the segment that starts there
    sig = merged_signals()
    tree = cKDTree(sig)
    n_signals = np.zeros(n_nodes)
    mid = (a + b) / 2
    for i in range(n_nodes):
        cand = tree.query_ball_point(mid[i], seg_len[i] / 2 + SIGNAL_RADIUS_M)
        if not cand:
            continue
        v = sig[cand] - a[i]
        t = (v @ unit[i]) / max(seg_len[i], 1e-6)
        perp = np.abs(v[:, 0] * unit[i, 1] - v[:, 1] * unit[i, 0])
        n_signals[i] = np.sum((t >= 0) & (t < 1) & (perp <= SIGNAL_RADIUS_M))

    # bus lanes: sample the segment every STEP_M, match parallel lane points nearby
    lp, lh, lded, lsid, scheds = lane_points()
    ltree = cKDTree(lp)
    ded_frac = np.zeros(n_nodes)
    shared_frac = np.zeros(n_nodes)
    active = np.zeros((n_nodes, 168), dtype=np.float32)
    for i in range(n_nodes):
        k = max(int(seg_len[i] // STEP_M), 1)
        samples = a[i] + np.outer((np.arange(k) + 0.5) / k, seg[i])
        hits = ltree.query_ball_point(samples, LANE_RADIUS_M)
        any_ded = np.zeros(k, dtype=bool)
        any_sh = np.zeros(k, dtype=bool)
        on = np.zeros((k, 168), dtype=bool)
        for j, h in enumerate(hits):
            if not h:
                continue
            h = np.array(h)
            h = h[np.abs(lh[h] @ unit[i]) >= LANE_MIN_COS]
            if len(h) == 0:
                continue
            any_ded[j] = lded[h].any()
            any_sh[j] = (~lded[h]).any()
            on[j] = scheds[np.unique(lsid[h])].any(0)
        ded_frac[i], shared_frac[i] = any_ded.mean(), any_sh.mean()
        active[i] = on.mean(0)

    attrs = pl.DataFrame({
        "node": nodes["node"], "seg_len_m": seg_len, "n_signals": n_signals,
        "signals_per_km": n_signals / np.maximum(seg_len, 50.0) * 1000,
        "lane_ded_frac": ded_frac, "lane_shared_frac": shared_frac,
    })
    attrs.write_parquet(GRAPH / "node_attrs.parquet")
    np.save(GRAPH / "node_lane_active.npy", active)
    print(attrs.select(ATTRS).describe())
    print(f"segments with any bus lane: {((ded_frac + shared_frac) > 0).mean():.1%}; "
          f"mean active share Mon 8h {active[:, 8].mean():.3f}, Mon 3h {active[:, 3].mean():.3f}")


if __name__ == "__main__":
    main()
