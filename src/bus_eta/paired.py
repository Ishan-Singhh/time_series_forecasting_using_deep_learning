"""Paired daily-MAE tests between seed-averaged models (daily MAE averaged over seeds).

The final-protocol comparisons of docs/FINDINGS.md §4 and §5.4: `python -m bus_eta.paired`.
"""
import numpy as np, polars as pl
from scipy.stats import wilcoxon
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
Q = seeds("stgnn_mta_qsdrop")
run("qsdrop vs original", Q, seeds("stgnn_mta"))
run("qsdrop vs q", Q, seeds("stgnn_mta_q"))
run("qsdrop vs sdrop", Q, seeds("stgnn_mta_sdrop"))
run("qsdrop vs gwnet", Q, seeds("gwnet_mta"))
run("qsdrop vs dcrnn", Q, seeds("dcrnn_mta"))
run("qsdrop vs staeformer", Q, seeds("staeformer_mta"))
run("qsdrop vs a3tgcn", Q, seeds("a3tgcn_mta"))
run("qsdrop vs tgcn", Q, ["tgcn_mta"])
run("qsdrop vs seq", Q, seeds("seq_mta"))
run("qsdrop vs convlstm e20", Q, seeds("convlstm_mta_e20"))
run("qsdrop vs lgbm", Q, ["lgbm_mta"])
run("qsdrop vs hist_avg", Q, ["hist_avg"])
run("convlstm e20 vs lgbm", seeds("convlstm_mta_e20"), ["lgbm_mta"])
run("gwnet vs seq", seeds("gwnet_mta"), seeds("seq_mta"))
run("stgnn_mta vs seq_mta", seeds("stgnn_mta"), seeds("seq_mta"))
run("stgnn_mta vs stgnn_no_mta", seeds("stgnn_mta"), seeds("stgnn_no_mta"))
run("seq_mta vs seq_no_mta", seeds("seq_mta"), seeds("seq_no_mta"))
run("stgnn_no_mta vs seq_no_mta", seeds("stgnn_no_mta"), seeds("seq_no_mta"))
run("lgbm_mta vs lgbm_no_mta", ["lgbm_mta"], ["lgbm_no_mta"])
run("staeformer vs gwnet", seeds("staeformer_mta"), seeds("gwnet_mta"))
run("staeformer vs dcrnn", seeds("staeformer_mta"), seeds("dcrnn_mta"))
run("staeformer vs a3tgcn", seeds("staeformer_mta"), seeds("a3tgcn_mta"))
run("staeformer vs seq", seeds("staeformer_mta"), seeds("seq_mta"))
run("staeformer vs lgbm", seeds("staeformer_mta"), ["lgbm_mta"])
I = seeds("analysis/stgnn_mta_v2indqsd_ho_abl")
run("cold: indqsd vs ind", I, seeds("analysis/stgnn_mta_v2ind_ho_abl"), HO)
run("cold: indqsd vs seq_ho_abl", I, seeds("analysis/seq_mta_ho_abl"), HO)
run("cold: indqsd vs lgbm_ho", I, ["analysis/lgbm_mta_ho"], HO)
run("cold: indqsd vs stgnn_ho_abl", I, ["analysis/stgnn_mta_ho_abl"], HO)
