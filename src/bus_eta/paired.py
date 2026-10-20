"""Paired daily-MAE tests between seed-averaged models (daily MAE averaged over seeds).

Main model: stgnn_mta_q (quantile head). stgnn_mta_qsdrop (+ observation dropout) is the
robust variant, and the inductive encoder (v2ind) the cold-start variant.

The final-protocol comparisons of docs/FINDINGS.md §4 and §5.4: `python -m bus_eta.paired`.
Brooklyn (§5.10: replication and zero-shot transfer): `BUS_ETA_BOROUGH=brooklyn python -m bus_eta.paired`.
"""
import numpy as np, polars as pl
from scipy.stats import wilcoxon
from bus_eta.config import BOROUGH
from bus_eta.evaluate import load_common
HO = ["M102", "M7", "M14D+", "M5", "M15+", "M42"]
def seeds(m, n=3): return [m] + [f"{m}_s{k}" for k in range(1, n)]
def daily(df, ms):
    return df.group_by("service_date").agg(
        *[(pl.col(m) - pl.col("y")).abs().mean().alias(m) for m in ms]).sort("service_date")
def run(label, a, b, routes=None):
    df = load_common(a + b, routes); d = daily(df, a + b)
    da = d.select(a).to_numpy().mean(1); db = d.select(b).to_numpy().mean(1)
    diff = da - db
    print(f"{label}: {da.mean():.2f} vs {db.mean():.2f} d={diff.mean():+.2f} better {np.mean(diff<0):.0%} of {len(diff)}d "
          f"p={wilcoxon(diff).pvalue:.1g}", flush=True)
M = seeds("stgnn_mta_q")
A = lambda m, n=3: [f"analysis/{x}" for x in seeds(m, n)]
if BOROUGH == "brooklyn":
    R = seeds("stgnn_mta_qsdrop")
    for label, b in [("original", seeds("stgnn_mta")), ("qsdrop", R), ("gwnet", seeds("gwnet_mta")),
                     ("seq", seeds("seq_mta")), ("lgbm", ["lgbm_mta"]), ("hist_avg", ["hist_avg"]), ("mta", ["mta"])]:
        run(f"q vs {label}", M, b)
    run("qsdrop vs gwnet", R, seeds("gwnet_mta"))
    run("qsdrop vs seq", R, seeds("seq_mta"))
    run("stgnn_mta vs gwnet", seeds("stgnn_mta"), seeds("gwnet_mta"))
    run("gwnet vs seq", seeds("gwnet_mta"), seeds("seq_mta"))
    run("stgnn_mta vs seq_mta", seeds("stgnn_mta"), seeds("seq_mta"))
    run("lgbm_mta vs lgbm_no_mta", ["lgbm_mta"], ["lgbm_no_mta"])
    run("randsplit lgbm vs q", ["analysis/lgbm_mta_randsplit"], M)
    # zero-shot: Manhattan cold-start checkpoints, no Brooklyn training
    XI, XQ, XS = A("xfer_stgnn_mta_v2ind_ho_abl"), A("xfer_stgnn_mta_v2indqsd_ho_abl"), A("xfer_seq_mta_ho_abl")
    run("xfer: ind vs seq", XI, XS)
    run("xfer: ind vs indqsd", XI, XQ)
    for label, b in [("mta", ["mta"]), ("mta_cal", ["mta_cal"]), ("hist_avg", ["hist_avg"]), ("lgbm (trained here)", ["lgbm_mta"]),
                     ("q (trained here)", M)]:
        run(f"xfer: seq vs {label}", XS, b)
        run(f"xfer: ind vs {label}", XI, b)
    raise SystemExit
run("q vs original", M, seeds("stgnn_mta"))
run("q vs qsdrop", M, seeds("stgnn_mta_qsdrop"))
run("q vs gwnet", M, seeds("gwnet_mta"))
run("q vs gwnet_q", M, seeds("gwnet_mta_q"))
run("q vs dcrnn", M, seeds("dcrnn_mta"))
run("q vs staeformer", M, seeds("staeformer_mta"))
run("q vs a3tgcn", M, seeds("a3tgcn_mta"))
run("q vs tgcn", M, ["tgcn_mta"])
run("q vs seq", M, seeds("seq_mta"))
run("q vs convlstm e20", M, seeds("convlstm_mta_e20"))
run("q vs lgbm", M, ["lgbm_mta"])
run("q vs hist_avg", M, ["hist_avg"])
R = seeds("stgnn_mta_qsdrop")
run("qsdrop vs original", R, seeds("stgnn_mta"))
run("qsdrop vs sdrop", R, seeds("stgnn_mta_sdrop"))
run("qsdrop vs gwnet", R, seeds("gwnet_mta"))
run("qsdrop vs dcrnn", R, seeds("dcrnn_mta"))
run("qsdrop vs seq", R, seeds("seq_mta"))
run("qsdrop vs lgbm", R, ["lgbm_mta"])
run("sdrop vs original", seeds("stgnn_mta_sdrop"), seeds("stgnn_mta"))
run("stgnn_mta vs gwnet", seeds("stgnn_mta"), seeds("gwnet_mta"))
run("stgnn_mta vs dcrnn", seeds("stgnn_mta"), seeds("dcrnn_mta"))
run("gwnet vs dcrnn", seeds("gwnet_mta"), seeds("dcrnn_mta"))
run("gwnet_q vs gwnet", seeds("gwnet_mta_q"), seeds("gwnet_mta"))
run("gwnet_q vs stgnn_mta", seeds("gwnet_mta_q"), seeds("stgnn_mta"))
run("qsdrop vs gwnet_q", R, seeds("gwnet_mta_q"))
run("convlstm e20 vs lgbm", seeds("convlstm_mta_e20"), ["lgbm_mta"])
run("gwnet vs seq", seeds("gwnet_mta"), seeds("seq_mta"))
run("stgnn_mta vs seq_mta", seeds("stgnn_mta"), seeds("seq_mta"))
run("stgnn_mta vs stgnn_no_mta", seeds("stgnn_mta"), seeds("stgnn_no_mta"))
run("seq_mta vs seq_no_mta", seeds("seq_mta"), seeds("seq_no_mta"))
run("stgnn_no_mta vs seq_no_mta", seeds("stgnn_no_mta"), seeds("seq_no_mta"))
run("lgbm_mta vs lgbm_no_mta", ["lgbm_mta"], ["lgbm_no_mta"])
run("staeformer vs seq", seeds("staeformer_mta"), seeds("seq_mta"))
run("staeformer vs lgbm", seeds("staeformer_mta"), ["lgbm_mta"])
run("randsplit lgbm vs q", ["analysis/lgbm_mta_randsplit"], M)
I = A("stgnn_mta_v2ind_ho_abl")
run("cold: ind vs indqsd", I, A("stgnn_mta_v2indqsd_ho_abl"), HO)
run("cold: ind vs seq_ho_abl", I, A("seq_mta_ho_abl"), HO)
run("cold: ind vs lgbm_ho", I, ["analysis/lgbm_mta_ho"], HO)
run("cold: ind vs stgnn_ho_abl", I, A("stgnn_mta_ho_abl"), HO)
G = A("gwnet_mta_ho_abl")
run("cold: ind vs gwnet_ho_abl", I, G, HO)
run("cold: gwnet_ho_abl vs stgnn_ho_abl", G, A("stgnn_mta_ho_abl"), HO)
run("cold: gwnet_ho_abl vs seq_ho_abl", G, A("seq_mta_ho_abl"), HO)
run("cold: gwnet_ho_abl vs gwnet trained with them", G, seeds("gwnet_mta"), HO)
run("cold: ind vs q trained with them", I, M, HO)
