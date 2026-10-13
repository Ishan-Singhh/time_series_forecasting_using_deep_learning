"""Corridor graph over physical bus segments, shared by every route that uses them.

Node  = a directed stop pair (from_stop -> to_stop) that is consecutive in some pattern
        of any schedule pick. Routes serving the same pair share the node.
Edges = "flow":     segment (a->b) to (b->c) consecutive in some pattern (directed)
        "corridor": segments whose midpoints are within CORRIDOR_RADIUS_M and whose
                    headings differ by < CORRIDOR_MAX_ANGLE_DEG: the same street and
                    direction served by different stop patterns (e.g. local vs SBS)

Outputs data/interim/graph/{nodes,edges}.parquet

    python -m bus_eta.graph
"""
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from bus_eta.config import INTERIM

STATIC = INTERIM / "static"
OUT = INTERIM / "graph"
CORRIDOR_RADIUS_M = 150.0
CORRIDOR_MAX_ANGLE_DEG = 30.0


def main() -> None:
    st = pl.read_parquet(STATIC / "stop_times.parquet", columns=["pick", "trip_id", "stop_idx", "stop_id", "dist_m"])
    trips = pl.read_parquet(STATIC / "trips.parquet", columns=["pick", "trip_id", "route_id", "pattern_id"])
    stops = pl.read_parquet(STATIC / "stops.parquet", columns=["stop_id", "x", "y"]).unique("stop_id")

    # one representative trip per pattern is enough to enumerate its stop pairs
    rep = trips.unique("pattern_id", keep="first")
    seq = (st.join(rep, on=["pick", "trip_id"]).sort("pattern_id", "stop_idx")
           .with_columns(pl.col("stop_id").shift(-1).over("pattern_id").alias("to_stop"),
                         pl.col("dist_m").shift(-1).over("pattern_id").alias("to_dist"))
           .drop_nulls("to_stop")
           .rename({"stop_id": "from_stop"}))

    nodes = (
        seq.group_by("from_stop", "to_stop").agg(
            (pl.col("to_dist") - pl.col("dist_m")).median().alias("length_m"),
            pl.col("route_id").unique().sort().alias("routes"))
        .sort("from_stop", "to_stop").with_row_index("node")
        .join(stops.rename({"stop_id": "from_stop", "x": "x0", "y": "y0"}), on="from_stop")
        .join(stops.rename({"stop_id": "to_stop", "x": "x1", "y": "y1"}), on="to_stop")
        .with_columns(((pl.col("x0") + pl.col("x1")) / 2).alias("mx"), ((pl.col("y0") + pl.col("y1")) / 2).alias("my"),
                      pl.arctan2(pl.col("y1") - pl.col("y0"), pl.col("x1") - pl.col("x0")).alias("heading"),
                      pl.col("routes").list.len().alias("n_routes"))
    )
    node_of = nodes.select("from_stop", "to_stop", "node")

    flow = (
        seq.join(node_of, on=["from_stop", "to_stop"])
        .with_columns(pl.col("node").shift(-1).over("pattern_id").alias("dst"))
        .drop_nulls("dst").select(pl.col("node").alias("src"), "dst").unique()
        .with_columns(pl.lit("flow").alias("edge_type"))
    )

    xy = nodes.select("mx", "my").to_numpy()
    heading = nodes["heading"].to_numpy()
    pairs = cKDTree(xy).query_pairs(CORRIDOR_RADIUS_M, output_type="ndarray")
    dh = np.abs(np.angle(np.exp(1j * (heading[pairs[:, 0]] - heading[pairs[:, 1]]))))
    pairs = pairs[dh < np.radians(CORRIDOR_MAX_ANGLE_DEG)]
    corridor = (
        pl.DataFrame({"src": np.concatenate([pairs[:, 0], pairs[:, 1]]),
                      "dst": np.concatenate([pairs[:, 1], pairs[:, 0]])}, schema={"src": pl.UInt32, "dst": pl.UInt32})
        .join(flow.select("src", "dst"), on=["src", "dst"], how="anti")
        .with_columns(pl.lit("corridor").alias("edge_type"))
    )
    edges = pl.concat([flow.with_columns(pl.col("src").cast(pl.UInt32), pl.col("dst").cast(pl.UInt32)), corridor])

    OUT.mkdir(parents=True, exist_ok=True)
    nodes.drop("routes").write_parquet(OUT / "nodes.parquet")
    edges.write_parquet(OUT / "edges.parquet")
    print(f"nodes {nodes.height:,} (shared by >1 route: {(nodes['n_routes'] > 1).sum():,}); "
          f"edges {edges.group_by('edge_type').len().sort('edge_type').rows()}")


if __name__ == "__main__":
    main()
