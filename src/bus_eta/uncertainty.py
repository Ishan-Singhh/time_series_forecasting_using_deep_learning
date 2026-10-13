"""Score predictive intervals from quantile models (stgnn --quantiles 0.1 0.5 0.9).

Per horizon band: empirical coverage of the central interval [q_lo, q_hi] (nominal
q_hi - q_lo), mean and median interval width, share of targets below each quantile
(calibration), mean pinball loss per quantile and the share of crossed quantiles.

    python -m bus_eta.uncertainty --models stgnn_mta_q
"""
import argparse

import numpy as np
import polars as pl

from bus_eta.evaluate import BANDS, KEY, PREDS, RESULTS, band_label, unique_keys


def qcols(path) -> list[str]:
    return sorted((c for c in pl.read_parquet_schema(path) if c[0] == "q" and c[1:].isdigit()), key=lambda c: int(c[1:]))


def score(name: str) -> pl.DataFrame:
    path = PREDS / f"{name}.parquet"
    qs = qcols(path)
    assert len(qs) >= 2, f"{name} has no quantile columns"
    truth = unique_keys(pl.read_parquet(PREDS / "_truth.parquet", columns=KEY + ["y"]))
    df = unique_keys(pl.read_parquet(path, columns=KEY + qs)).join(truth, on=KEY).drop_nulls()
    lo, hi = qs[0], qs[-1]
    rows = []
    for a, b in BANDS:
        sub = df.filter(pl.col("h").is_between(a, b))
        y = sub["y"].to_numpy()
        q = {c: sub[c].to_numpy() for c in qs}
        width = q[hi] - q[lo]
        r = {"model": name, "band": band_label(a, b), "n": len(y),
             f"nominal_{lo}-{hi}_%": int(hi[1:]) - int(lo[1:]),
             "coverage_%": 100 * np.mean((y >= q[lo]) & (y <= q[hi])),
             "width_mean_s": width.mean(), "width_median_s": np.median(width),
             "crossed_%": 100 * np.mean(np.any(np.diff(np.stack([q[c] for c in qs]), axis=0) < 0, axis=0))}
        for c in qs:
            t = int(c[1:]) / 100
            e = y - q[c]
            r[f"below_{c}_%"] = 100 * np.mean(y < q[c])
            r[f"pinball_{c}"] = np.mean(np.maximum(t * e, (t - 1) * e))
        rows.append(r)
    return pl.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--name", default="uncertainty")
    args = p.parse_args()
    res = pl.concat([score(m) for m in args.models], how="diagonal")
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.write_csv(RESULTS / f"{args.name}.csv")
    with pl.Config(tbl_rows=60, tbl_cols=30, tbl_width_chars=250, float_precision=2):
        print(res)


if __name__ == "__main__":
    main()
