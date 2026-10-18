"""Conformal recalibration of the quantile intervals (split CQR, Romano et al. 2019).

For each central interval [q_lo, q_hi] of a quantile model (50, 80, 90%) and each horizon h,
the conformity score on the validation month is s = max(q_lo - y, y - q_hi); the interval is
widened (or narrowed, if s is mostly negative) by the ceil((n + 1)(1 - a)) / n quantile of s.
Calibrating per horizon keeps coverage per horizon, not only on average. The test month is
then scored before and after: coverage, mean width and the interval (Winkler) score
width + (2 / a) * miss distance, which is proper, so it rewards calibration and sharpness.

Validation predictions come from `stgnn --eval-only <model> --split val --name <model>_val`.

    python -m bus_eta.conformal --models stgnn_mta_q stgnn_mta_q_s1 stgnn_mta_q_s2
"""
import argparse

import numpy as np
import polars as pl

from bus_eta.evaluate import BANDS, KEY, PREDS, RESULTS, band_label, unique_keys

PAIRS = [("q25", "q75"), ("q10", "q90"), ("q5", "q95")]


def offsets(val: pl.DataFrame, lo: str, hi: str, alpha: float) -> dict[int, float]:
    """Per-horizon CQR offset from the validation scores."""
    out = {}
    for (h,), g in val.group_by("h"):
        s = np.maximum(g[lo] - g["y"], g["y"] - g[hi]).to_numpy()
        n = len(s)
        out[h] = float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
    return out


def score(name: str) -> pl.DataFrame:
    val = unique_keys(pl.read_parquet(PREDS / "analysis" / f"{name}_val.parquet")).drop_nulls().filter(pl.col("y").is_not_nan())
    truth = unique_keys(pl.read_parquet(PREDS / "_truth.parquet", columns=KEY + ["y"]))
    cols = sorted({c for p in PAIRS for c in p})
    test = unique_keys(pl.read_parquet(PREDS / f"{name}.parquet", columns=KEY + cols)).join(truth, on=KEY).drop_nulls()
    rows = []
    for lo, hi in PAIRS:
        nominal = int(hi[1:]) - int(lo[1:])
        alpha = 1 - nominal / 100
        off = offsets(val, lo, hi, alpha)
        t = test.with_columns(pl.col("h").replace_strict(off, return_dtype=pl.Float64).alias("_off"))
        for kind, l, u in (("raw", pl.col(lo), pl.col(hi)),
                           ("conformal", pl.col(lo) - pl.col("_off"), pl.col(hi) + pl.col("_off"))):
            t2 = t.with_columns(l.alias("_l"), u.alias("_u"))
            for a, b in BANDS:
                sub = t2.filter(pl.col("h").is_between(a, b))
                y, L, U = sub["y"].to_numpy(), sub["_l"].to_numpy(), sub["_u"].to_numpy()
                w = U - L
                miss = np.maximum(L - y, 0) + np.maximum(y - U, 0)
                v = val.filter(pl.col("h").is_between(a, b))
                rows.append({"model": name, "nominal_%": nominal, "kind": kind, "band": band_label(a, b),
                             "n": len(y), "coverage_%": 100 * np.mean(miss == 0), "width_mean_s": w.mean(),
                             "interval_score_s": np.mean(w + 2 / alpha * miss),
                             "val_raw_coverage_%": 100 * np.mean((v["y"] >= v[lo]) & (v["y"] <= v[hi])),
                             "offset_mean_s": float(np.mean([off[h] for h in range(a, b + 1)]))})
    return pl.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--name", default="conformal")
    args = p.parse_args()
    res = pl.concat([score(m) for m in args.models])
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.write_csv(RESULTS / f"{args.name}.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=250, float_precision=2):
        print(res.filter(pl.col("band") == "all"))
        # seed-averaged, base model name = before the _s<seed> suffix
        avg = (res.with_columns(pl.col("model").str.replace(r"_s\d+$", "").alias("model"))
               .group_by("model", "nominal_%", "kind", "band", maintain_order=True)
               .agg(pl.col("coverage_%", "width_mean_s", "interval_score_s", "offset_mean_s").mean()))
        avg.write_csv(RESULTS / f"{args.name}_mean.csv")
        print(avg)


if __name__ == "__main__":
    main()
