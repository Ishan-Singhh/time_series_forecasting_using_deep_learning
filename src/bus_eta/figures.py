"""Paper figures. Each figure's numbers are computed once from the prediction files on the
common test support and cached as a small CSV in results/figdata/; plotting reads the
cache, so restyling needs no recomputation (delete a CSV, or pass --recompute, to rebuild).

    python -m bus_eta.figures                    # all figures -> docs/figures/*.pdf, *.png
    python -m bus_eta.figures horizon tod        # some of them
    python -m bus_eta.figures --recompute map    # rebuild the cached numbers first

Figures:
  horizon     MAE and skill vs MTA by horizon h = 1..20 (main baselines, mean over seeds)
  robustness  MAE vs share of network observations hidden at test time
  permutation MAE increase when one input group is shuffled on test
  tod         MAE and graph gain by hour of day
  map         the segment graph of Manhattan, coloured by the graph gain per segment
  trip        one worked departure: actual vs predicted arrival times with 90% interval
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection

from bus_eta.config import INTERIM
from bus_eta.evaluate import KEY, PREDS, RESULTS, load_common

FIGS = RESULTS.parents[2] / "docs" / "figures"
CACHE = RESULTS / "figdata"

MAIN = "stgnn_mta_q"          # the paper's main model (quantile head)
ROBUST_VARIANT = "stgnn_mta_qsdrop"  # + observation dropout

# Okabe-Ito, colour-blind safe; ours is the strong vermillion, MTA black
STYLE = {
    MAIN: ("Ours", "#D55E00", "-", 2.2),
    "gwnet_mta": ("Graph WaveNet", "#0072B2", "-", 1.3),
    "dcrnn_mta": ("DCRNN", "#56B4E9", "-", 1.3),
    "a3tgcn_mta": ("A3T-GCN", "#009E73", "-", 1.3),
    "staeformer_mta": ("STAEformer", "#CC79A7", "-", 1.3),
    "seq_mta": ("No graph", "#E69F00", "--", 1.5),
    "convlstm_mta_e20": ("ConvLSTM", "#999999", "-", 1.3),
    "lgbm_mta": ("LightGBM", "#A08C00", "-", 1.5),
    "hist_avg": ("Historical average", "#666666", ":", 1.5),
    "mta_cal": ("MTA (calibrated)", "#000000", "-", 1.5),
}
SEEDED = {MAIN, ROBUST_VARIANT, "gwnet_mta", "dcrnn_mta", "a3tgcn_mta", "staeformer_mta", "seq_mta",
          "convlstm_mta_e20"}


def seeds(m: str) -> list[str]:
    return [m, f"{m}_s1", f"{m}_s2"] if m in SEEDED else [m]


def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#DDDDDD", "grid.linewidth": 0.5, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote docs/figures/{name}.pdf / .png", flush=True)


def cached(name: str, build, recompute: bool) -> pl.DataFrame:
    f = CACHE / f"{name}.csv"
    if recompute or not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"computing {name} ...", flush=True)
        build().write_csv(f)
    return pl.read_csv(f)


def seed_mean_abs(models: list[str]) -> list[pl.Expr]:
    """Absolute error of each model, averaged over its seeds (one column per model)."""
    return [pl.mean_horizontal([(pl.col(s) - pl.col("y")).abs() for s in seeds(m)]).alias(m) for m in models]


# ---------------------------------------------------------------- horizon
def build_horizon() -> pl.DataFrame:
    cols = [s for m in STYLE for s in seeds(m)]
    df = load_common(cols)
    return (df.with_columns(seed_mean_abs(list(STYLE))).group_by("h")
            .agg(pl.len().alias("n"), *[pl.col(m).mean() for m in STYLE]).sort("h"))


def plot_horizon(d: pl.DataFrame) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    h = d["h"].to_numpy()
    ref = d["mta_cal"].to_numpy()
    for m, (label, c, ls, lw) in STYLE.items():
        y = d[m].to_numpy()
        a.plot(h, y, ls, color=c, lw=lw, label=label, zorder=3 if m == MAIN else 2)
        if m != "mta_cal":
            b.plot(h, 100 * (1 - y / ref), ls, color=c, lw=lw, zorder=3 if m == MAIN else 2)
    b.axhline(0, color="black", lw=0.8)
    a.set(xlabel="Horizon h (stops ahead)", ylabel="MAE (s)", xticks=[1, 5, 10, 15, 20])
    b.set(xlabel="Horizon h (stops ahead)", ylabel="Skill vs MTA calibrated (%)", xticks=[1, 5, 10, 15, 20])
    a.set_title("(a) Error grows with horizon")
    b.set_title("(b) Skill vs MTA")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "horizon")


# ---------------------------------------------------------------- robustness
DROPS = [0, 25, 50, 75, 90]
ROBUST = {MAIN: ("Ours", "#D55E00", "o--"),
          "gwnet_mta": ("Graph WaveNet", "#0072B2", "o--"),
          ROBUST_VARIANT: ("Ours + obs. dropout in training", "#D55E00", "s-")}


def build_robustness() -> pl.DataFrame:
    cols = [f"analysis/{m}_drop{p}" for m in ROBUST for p in DROPS] + seeds("seq_mta") + ["mta_cal"]
    df = load_common(cols).with_columns(seed_mean_abs(["seq_mta"]))
    rows = []
    for m in ROBUST:
        for p in DROPS:
            rows.append({"model": m, "hidden": p,
                         "MAE": df.select((pl.col(f"analysis/{m}_drop{p}") - pl.col("y")).abs().mean()).item()})
    for m in ["seq_mta", "mta_cal"]:
        mae = df["seq_mta"].mean() if m == "seq_mta" else df.select((pl.col(m) - pl.col("y")).abs().mean()).item()
        rows += [{"model": m, "hidden": p, "MAE": mae} for p in DROPS]
    return pl.DataFrame(rows)


def plot_robustness(d: pl.DataFrame) -> None:
    fig, a = plt.subplots(figsize=(3.4, 2.4))
    for m, (label, c, fmt) in ROBUST.items():
        s = d.filter(pl.col("model") == m).sort("hidden")
        a.plot(s["hidden"], s["MAE"], fmt, color=c, lw=1.6, ms=3.5, label=label)
    seq = d.filter(pl.col("model") == "seq_mta")["MAE"][0]
    a.axhline(seq, color="#E69F00", ls="--", lw=1.2, label=f"No graph ({seq:.1f} s)")
    a.set(xlabel="Network observations hidden at test time (%)", ylabel="MAE (s)", xticks=DROPS)
    a.legend(frameon=False, loc="upper left")
    save(fig, "robustness")


# ---------------------------------------------------------------- permutation
PERM = [("rt", "Real-time segment estimate"), ("schedule", "Schedule / distances"),
        ("graph", "Network state (graph)"), ("bus_state", "Bus state"), ("mta", "MTA prediction"),
        ("time", "Time of day"), ("route", "Route id"), ("dow", "Day of week"),
        ("calendar_weather", "Weather + calendar")]


def build_permutation() -> pl.DataFrame:
    cols = ["stgnn_mta"] + [f"analysis/stgnn_mta_perm_{g}" for g, _ in PERM]
    df = load_common(cols)
    base = df.select((pl.col("stgnn_mta") - pl.col("y")).abs().mean()).item()
    return pl.DataFrame([{"group": g, "label": lab, "base": base,
                          "MAE": df.select((pl.col(f"analysis/stgnn_mta_perm_{g}") - pl.col("y")).abs().mean()).item()}
                         for g, lab in PERM]).with_columns((pl.col("MAE") - pl.col("base")).alias("delta"))


def plot_permutation(d: pl.DataFrame) -> None:
    d = d.sort("delta")
    fig, a = plt.subplots(figsize=(3.4, 2.5))
    col = ["#D55E00" if g == "graph" else "#0072B2" for g in d["group"]]
    a.barh(d["label"], d["delta"], color=col, height=0.65)
    a.set_xscale("log")
    a.set_xlim(0.05, 1000)
    for i, v in enumerate(d["delta"]):
        a.text(v * 1.15, i, f"+{v:.1f} s" if v < 10 else f"+{v:.0f} s", va="center", fontsize=6.5)
    a.set_xlabel(f"MAE increase when shuffled (s, log scale; base {d['base'][0]:.1f} s)")
    a.grid(axis="y", visible=False)
    save(fig, "permutation")


# ---------------------------------------------------------------- time of day
TOD = [MAIN, "seq_mta", "lgbm_mta", "mta_cal"]


def build_tod() -> pl.DataFrame:
    cols = [s for m in TOD for s in seeds(m)]
    df = load_common(cols).with_columns(*seed_mean_abs(TOD), pl.col("tod_hour").floor().cast(pl.Int8).alias("hour"))
    return df.group_by("hour").agg(pl.len().alias("n"), *[pl.col(m).mean() for m in TOD]).sort("hour")


def plot_tod(d: pl.DataFrame) -> None:
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 3.3), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    hr = d["hour"].to_numpy()
    for m in TOD:
        label, c, ls, lw = STYLE[m]
        a.plot(hr, d[m], ls, color=c, lw=lw, label=label)
    a.set_ylabel("MAE (s)")
    a.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
             handlelength=1.5)
    b.bar(hr, d["seq_mta"] - d[MAIN], color="#D55E00", width=0.75)
    b.set(xlabel="Hour of day (issue time)", ylabel="Graph gain (s)", xticks=range(0, 24, 3))
    save(fig, "tod")


# ---------------------------------------------------------------- map
MIN_N_SEG = 500


def build_map() -> pl.DataFrame:
    """Graph gain per segment: no-graph minus ours (seed-averaged absolute error) over every
    prediction whose target stop k + h is reached through that segment, all horizons."""
    from bus_eta.features import stop_segments
    cols = seeds(MAIN) + seeds("seq_mta")
    df = (load_common(cols).with_columns(seed_mean_abs([MAIN, "seq_mta"]))
          .with_columns((pl.col("k") + pl.col("h")).cast(pl.Int32).alias("stop_idx"))
          .select("trip_id", "stop_idx", MAIN, "seq_mta"))
    seg = stop_segments().with_columns(pl.col("stop_idx").cast(pl.Int32)).collect()
    per = (df.join(seg, on=["trip_id", "stop_idx"], how="inner").group_by("node")
           .agg(pl.len().alias("n"), pl.col(MAIN).mean().alias("ours"),
                pl.col("seq_mta").mean().alias("seq")))
    nodes = pl.read_parquet(INTERIM / "graph" / "nodes.parquet").select("node", "x0", "y0", "x1", "y1", "n_routes")
    return (nodes.with_columns(pl.col("node").cast(pl.Int64)).join(per, on="node", how="left")
            .with_columns((pl.col("seq") - pl.col("ours")).alias("gain")))


def plot_map(d: pl.DataFrame) -> None:
    fig, a = plt.subplots(figsize=(3.4, 5.2))
    th = np.deg2rad(29)  # Manhattan's street grid runs 29 degrees east of true north
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])  # turn it upright
    seg = lambda f: np.stack([f.select("x0", "y0").to_numpy() @ rot.T,
                              f.select("x1", "y1").to_numpy() @ rot.T], axis=1) / 1000
    few = d.filter(pl.col("n").is_null() | (pl.col("n") < MIN_N_SEG))
    ok = d.filter(pl.col("n") >= MIN_N_SEG).sort("gain")
    a.add_collection(LineCollection(seg(few), colors="#CCCCCC", linewidths=0.6))
    lim = float(np.quantile(np.abs(ok["gain"].to_numpy()), 0.95))
    lc = LineCollection(seg(ok), cmap="RdBu", linewidths=1.4)
    lc.set_array(ok["gain"].to_numpy())
    lc.set_clim(-lim, lim)
    a.add_collection(lc)
    a.autoscale()
    a.set_aspect("equal")
    a.set_axis_off()
    x0, y0 = a.get_xlim()[0], a.get_ylim()[0]
    a.plot([x0, x0 + 2], [y0, y0], color="black", lw=1.5)
    a.text(x0 + 1, y0 + 0.3, "2 km", ha="center", fontsize=7)
    a.text(x0, a.get_ylim()[1], "Manhattan north \u2191", fontsize=7, va="top")
    cb = fig.colorbar(lc, ax=a, shrink=0.6, pad=0.02)
    cb.set_label("Graph gain (s): no-graph MAE minus ours")
    pos = (ok["gain"] > 0).mean()
    a.set_title(f"{ok.height:,} segments with \u2265{MIN_N_SEG} predictions; graph helps on {pos:.0%}")
    save(fig, "map")


# ---------------------------------------------------------------- worked trip
def build_trip() -> pl.DataFrame:
    """A weekday PM-peak departure with all 20 horizons whose error for ours is near the
    median: actual and predicted travel time to each of the next 20 stops."""
    q = ["q5", "q50", "q95"]
    ours = pl.read_parquet(PREDS / f"{MAIN}.parquet", columns=KEY + ["pred"] + q)
    df = (load_common([MAIN, "mta_cal", "seq_mta"])
          .filter(pl.col("dow") <= 5, pl.col("tod_hour").is_between(16, 19)))
    full = df.group_by("trip_id", "start_date", "k").agg(pl.len().alias("n"),
                                                          (pl.col(MAIN) - pl.col("y")).abs().mean().alias("err"))
    full = full.filter(pl.col("n") == 20)
    med = full["err"].median()
    pick = full.with_columns((pl.col("err") - med).abs().alias("d")).sort("d", "trip_id").head(1)
    out = df.join(pick.select("trip_id", "start_date", "k"), on=["trip_id", "start_date", "k"])
    return out.join(ours.select(KEY + q), on=KEY).sort("h").select(
        "trip_id", "start_date", "k", "route_id", "service_date", "tod_hour", "h", "y",
        MAIN, "seq_mta", "mta_cal", *q)


def plot_trip(d: pl.DataFrame) -> None:
    fig, a = plt.subplots(figsize=(3.4, 2.5))
    h = d["h"].to_numpy()
    m = lambda c: d[c].to_numpy() / 60
    a.fill_between(h, m("q5"), m("q95"), color="#D55E00", alpha=0.18, lw=0, label="Ours, 90% interval")
    a.plot(h, m("y"), "o", color="black", ms=3, label="Actual (GPS)", zorder=4)
    a.plot(h, m(MAIN), "-", color="#D55E00", lw=1.8, label="Ours")
    a.plot(h, m("mta_cal"), "-", color="#000000", lw=1.0, label="MTA (calibrated)")
    a.plot(h, m("seq_mta"), "--", color="#E69F00", lw=1.2, label="No graph")
    r = d.row(0, named=True)
    hh, mm = divmod(int(round(r["tod_hour"] * 60)), 60)
    a.set(xlabel="Stops ahead", ylabel="Travel time from now (min)", xticks=[1, 5, 10, 15, 20],
          title=f"Route {r['route_id']}, {r['service_date']} {hh:02d}:{mm:02d}")
    a.legend(frameon=False, loc="upper left")
    save(fig, "trip")


FIGURES = {"horizon": (build_horizon, plot_horizon), "robustness": (build_robustness, plot_robustness),
           "permutation": (build_permutation, plot_permutation), "tod": (build_tod, plot_tod),
           "map": (build_map, plot_map), "trip": (build_trip, plot_trip)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("figures", nargs="*", help=f"any of {', '.join(FIGURES)} (default: all)")
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    if bad := set(args.figures) - set(FIGURES):
        p.error(f"unknown figure(s): {', '.join(sorted(bad))}")
    setup()
    for name in args.figures or FIGURES:
        build, plot = FIGURES[name]
        plot(cached(name, build, args.recompute))


if __name__ == "__main__":
    main()
