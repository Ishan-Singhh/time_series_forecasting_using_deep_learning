"""Network state tensor for the graph model: (time bin, segment node, feature).

Every observed traversal of a segment node (any route) is filed under the 5-minute bin in
which it was *revealed* (arr_seen of the downstream stop), so bin b only contains what a
real-time system knew by the end of b. A sample issued at t may use bins < t // BIN_S.

Features per (bin, node): mean speed (m/s), number of traversals, mean dwell at the
segment's upstream stop (s). Missing = NaN (no bus revealed anything in that bin).

Outputs data/processed/network_state/{state.npy (float16), meta.json}

    python -m bus_eta.network_state
"""
import json

import numpy as np
import polars as pl

from bus_eta.config import INTERIM, PROCESSED, SPLITS

ARRIVALS = INTERIM / "arrivals"
GRAPH = INTERIM / "graph"
OUT = PROCESSED / "network_state"
BIN_S = 300
MAX_GAP_S = 90.0
FEATURES = ["speed_mps", "n_obs", "dwell_s"]


def main() -> None:
    nodes = pl.read_parquet(GRAPH / "nodes.parquet", columns=["node", "from_stop", "to_stop", "length_m"])
    ev = pl.scan_parquet(str(ARRIVALS / "*.parquet")).select(
        "trip_id", "start_date", "stop_idx", "stop_id", "t_arr", "arr_gap_s", "arr_seen", "t_dep", "dep_gap_s"
    ).collect(engine="streaming").sort("trip_id", "start_date", "stop_idx")

    nxt = [pl.col(c).shift(-1).over("trip_id", "start_date") for c in ("stop_id", "stop_idx", "t_arr", "arr_gap_s", "arr_seen")]
    seg = (
        ev.with_columns(nxt[0].alias("to_stop"), nxt[1].alias("to_idx"), nxt[2].alias("to_arr"),
                        nxt[3].alias("to_gap"), nxt[4].alias("to_seen"))
        .filter((pl.col("to_idx") == pl.col("stop_idx") + 1) & (pl.col("dep_gap_s") <= MAX_GAP_S)
                & (pl.col("to_gap") <= MAX_GAP_S) & (pl.col("arr_gap_s") <= MAX_GAP_S))
        .join(nodes, left_on=["stop_id", "to_stop"], right_on=["from_stop", "to_stop"])
        .with_columns((pl.col("length_m") / (pl.col("to_arr") - pl.col("t_dep")).clip(lower_bound=1.0)).alias("speed_mps"),
                      (pl.col("t_dep") - pl.col("t_arr")).alias("dwell_s"),
                      (pl.col("to_seen") // BIN_S).cast(pl.Int64).alias("bin"))
        .filter(pl.col("speed_mps") < 30)
    )

    first = int(SPLITS["train"][0].strftime("%s")) // BIN_S - 288  # a day of margin for time zones
    last = int(SPLITS["test"][1].strftime("%s")) // BIN_S + 2 * 288
    agg = (seg.filter(pl.col("bin").is_between(first, last - 1))
           .group_by("bin", "node").agg(pl.col("speed_mps").mean(), pl.len().alias("n_obs"),
                                        pl.col("dwell_s").mean()))
    n_bins, n_nodes = last - first, nodes.height
    state = np.full((n_bins, n_nodes, len(FEATURES)), np.nan, dtype=np.float16)
    b = (agg["bin"].to_numpy() - first)
    n = agg["node"].to_numpy()
    for i, f in enumerate(FEATURES):
        state[b, n, i] = agg[f].to_numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "state.npy", state)
    (OUT / "meta.json").write_text(json.dumps({"first_bin": first, "bin_s": BIN_S, "features": FEATURES,
                                               "n_bins": n_bins, "n_nodes": n_nodes}))
    observed = ~np.isnan(state[..., 0])
    print(f"state {state.shape}; {seg.height:,} traversals; node-bins observed {observed.mean():.1%}; "
          f"nodes ever observed {observed.any(0).mean():.1%}")


if __name__ == "__main__":
    main()
