"""Planar projection and along-shape distance kernels.

All geometry is done in a local equirectangular projection (meters) centred on Manhattan,
which is accurate to well under 0.1% at city scale.
"""
import math

import numpy as np
from numba import njit

LAT0 = 40.78
LON0 = -73.97
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(LAT0))


def to_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = (np.asarray(lon, dtype=np.float64) - LON0) * M_PER_DEG_LON
    y = (np.asarray(lat, dtype=np.float64) - LAT0) * M_PER_DEG_LAT
    return x, y


def cumulative_distance(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    seg = np.hypot(np.diff(xs), np.diff(ys))
    return np.concatenate([[0.0], np.cumsum(seg)])


@njit(cache=True)
def _nearest_on_segments(px, py, xs, ys, cum, d_lo, d_hi):
    """Closest point to (px, py) on segments whose span overlaps [d_lo, d_hi].

    Returns (distance along shape, lateral offset). Lateral is inf if no segment qualifies.
    """
    best_lat = np.inf
    best_d = -1.0
    for i in range(len(xs) - 1):
        if cum[i + 1] < d_lo:
            continue
        if cum[i] > d_hi:
            break
        ax, ay = xs[i], ys[i]
        dx, dy = xs[i + 1] - ax, ys[i + 1] - ay
        L2 = dx * dx + dy * dy
        u = 0.0 if L2 == 0.0 else ((px - ax) * dx + (py - ay) * dy) / L2
        u = min(1.0, max(0.0, u))
        qx, qy = ax + u * dx, ay + u * dy
        lat = math.hypot(px - qx, py - qy)
        if lat < best_lat:
            best_lat = lat
            best_d = cum[i] + u * math.sqrt(L2)
    return best_d, best_lat


@njit(cache=True)
def project_stops(sx, sy, xs, ys, cum):
    """Distance along the shape for an ordered stop list, constrained to be non-decreasing.

    Each stop is searched only forward of the previous stop, up to a generous gap
    (3x straight-line distance + 500 m), so terminal loops can't send a stop backwards
    or ahead to a later pass over the same street.
    """
    n = len(sx)
    out = np.empty(n)
    lat_out = np.empty(n)
    prev = 0.0
    for k in range(n):
        if k == 0:
            hi = cum[-1]
        else:
            hi = prev + 3.0 * math.hypot(sx[k] - sx[k - 1], sy[k] - sy[k - 1]) + 500.0
        d, lat = _nearest_on_segments(sx[k], sy[k], xs, ys, cum, prev, hi)
        if lat == np.inf:
            d = prev
        out[k] = d
        lat_out[k] = lat
        prev = d
    return out, lat_out


@njit(cache=True)
def project_track(px, py, pt, a_lo, a_hi, xs, ys, cum, max_lateral, max_speed):
    """Project a time-ordered GPS track onto a shape, tracking forward progress.

    Until the first ping is accepted, the search is limited to the anchor window
    [a_lo, a_hi] (derived from the feed's next-stop field; nan = whole shape), which keeps
    a bus at a loop terminal from latching onto the end of its own shape. Later pings
    search from slightly behind the last accepted position to as far as max_speed allows.
    Pings further than max_lateral from the route are rejected (d = nan).
    """
    n = len(px)
    d_out = np.full(n, np.nan)
    lat_out = np.full(n, np.nan)
    have = False
    prev_d = 0.0
    prev_t = 0.0
    for k in range(n):
        if have:
            lo = prev_d - 50.0
            hi = prev_d + max_speed * (pt[k] - prev_t) + 100.0
        elif np.isnan(a_lo[k]):
            lo = 0.0
            hi = cum[-1]
        else:
            lo = a_lo[k]
            hi = a_hi[k]
        d, lat = _nearest_on_segments(px[k], py[k], xs, ys, cum, lo, hi)
        if lat <= max_lateral:
            d_out[k] = d
            lat_out[k] = lat
            prev_d = d
            prev_t = pt[k]
            have = True
    return d_out, lat_out
