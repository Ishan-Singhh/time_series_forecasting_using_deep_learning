"""Score every saved test prediction on a common support.

Predictions live in data/processed/preds/<model>.parquet with keys
(trip_id, start_date, k, h) and a `pred` column (seconds from t_issue to arrival).
Only rows where ALL compared models have a prediction are scored, so every model is
evaluated on identical rows. Metrics per horizon band: MAE (+ day-bootstrap 95% CI), MdAE,
RMSE, MAPE (y >= 60 s), WAPE, R2, P90 absolute error, median bias, % within 60/120 s and
skill vs calibrated MTA. A paired significance table compares each model with calibrated
MTA on daily MAE (bootstrap CI + Wilcoxon signed-rank), because samples within a day are
strongly correlated and days are the honest independent unit.

    python -m bus_eta.evaluate                       # all models in preds/
    python -m bus_eta.evaluate --models mta lgbm_no_mta lgbm_mta
"""
import argparse

import numpy as np
import polars as pl
from scipy.stats import wilcoxon

from bus_eta.config import PROCESSED

PREDS = PROCESSED / "preds"
RESULTS = PROCESSED / "results"
KEY = ["trip_id", "start_date", "k", "h"]
BANDS = [(1, 1), (2, 5), (6, 10), (11, 15), (16, 20), (1, 20)]
N_BOOT = 200
REFERENCE = "mta_cal"  # strongest non-learned baseline; skill and significance are vs this


def band_label(lo: int, hi: int) -> str:
    return "all" if (lo, hi) == (1, 20) else (f"h{lo}" if lo == hi else f"h{lo}-{hi}")


def unique_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Drop keys that occur more than once (two vehicles reported on the same trip_id):
    the issue is ambiguous, and repeated keys would multiply across the model joins."""
    return df.filter(pl.len().over(KEY) == 1)


def load_common(models: list[str], routes: list[str] | None = None) -> pl.DataFrame:
    df = unique_keys(pl.read_parquet(PREDS / "_truth.parquet"))
    if routes:
        df = df.filter(pl.col("route_id").is_in(routes))
    for m in models:
        p = unique_keys(pl.read_parquet(PREDS / f"{m}.parquet")).select(KEY + [pl.col("pred").alias(m)])
        df = df.join(p, on=KEY, how="inner")
    return df.drop_nulls(models)


def score(df: pl.DataFrame, models: list[str], seed: int = 0) -> pl.DataFrame:
    """Point-accuracy metrics per horizon band, with a day-bootstrap 95% CI on MAE."""
    days = df["service_date"].unique().sort()
    rng = np.random.default_rng(seed)
    boot_days = [days.gather(rng.integers(0, len(days), len(days))) for _ in range(N_BOOT)]
    rows = []
    for lo, hi in BANDS:
        sub = df.filter(pl.col("h").is_between(lo, hi))
        y = sub["y"].to_numpy()
        # per-day sums let the bootstrap resample days cheaply
        per_day = sub.group_by("service_date").agg(
            pl.len().alias("n"), *[(pl.col(m) - pl.col("y")).abs().sum().alias(m) for m in models])
        ref_mae = None
        if REFERENCE in models:
            ref_mae = np.abs(sub[REFERENCE].to_numpy() - y).mean()
        for m in models:
            ae = np.abs(sub[m].to_numpy() - y)
            e = sub[m].to_numpy() - y
            boot = []
            for bd in boot_days:
                pdays = pl.DataFrame({"service_date": bd}).join(per_day, on="service_date")
                boot.append(pdays[m].sum() / pdays["n"].sum())
            long_enough = y >= 60  # MAPE is unstable for very short trips
            rows.append({
                "band": band_label(lo, hi), "model": m, "n": sub.height,
                "MAE": ae.mean(), "MAE_lo": float(np.percentile(boot, 2.5)),
                "MAE_hi": float(np.percentile(boot, 97.5)),
                "MdAE": float(np.median(ae)),
                "RMSE": float(np.sqrt((e ** 2).mean())),
                "MAPE_%": 100 * float((ae[long_enough] / y[long_enough]).mean()),
                "WAPE_%": 100 * ae.sum() / y.sum(),
                "R2": 1 - float((e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                "P90_AE": float(np.percentile(ae, 90)),
                "bias_median": float(np.median(e)),
                "within_60s_%": 100 * float((ae <= 60).mean()),
                "within_120s_%": 100 * float((ae <= 120).mean()),
                "skill_vs_mta_%": None if ref_mae is None else 100 * (1 - ae.mean() / ref_mae),
            })
    return pl.DataFrame(rows)


def significance(df: pl.DataFrame, models: list[str], seed: int = 0) -> pl.DataFrame:
    """Paired test of each model against REFERENCE on daily MAE (days are the independent units).

    Reports the mean daily MAE difference (model - reference; negative = better), its
    day-bootstrap 95% CI, and a two-sided Wilcoxon signed-rank p-value.
    """
    if REFERENCE not in models:
        return pl.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for lo, hi in BANDS:
        daily = (df.filter(pl.col("h").is_between(lo, hi)).group_by("service_date")
                 .agg(*[(pl.col(m) - pl.col("y")).abs().mean().alias(m) for m in models]))
        ref = daily[REFERENCE].to_numpy()
        for m in models:
            if m == REFERENCE:
                continue
            diff = daily[m].to_numpy() - ref
            boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
            rows.append({
                "band": band_label(lo, hi), "model": m, "days": len(diff),
                "dMAE_vs_ref": float(diff.mean()), "dMAE_lo": float(np.percentile(boot, 2.5)),
                "dMAE_hi": float(np.percentile(boot, 97.5)),
                "days_better_%": 100 * float((diff < 0).mean()),
                "wilcoxon_p": float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
            })
    return pl.DataFrame(rows)


def seed_summary(res: pl.DataFrame) -> pl.DataFrame:
    """Mean and sd over seeds: <model>_s<k> files are extra seeds of <model> (seed 0)."""
    base = pl.col("model").str.replace(r"_s\d+$", "")
    metrics = ["MAE", "MdAE", "RMSE", "MAPE_%", "R2", "P90_AE", "within_60s_%", "skill_vs_mta_%"]
    return (res.with_columns(base.alias("model"))
            .group_by("band", "model")
            .agg(pl.len().alias("seeds"), *[pl.col(m).mean() for m in metrics],
                 *[pl.col(m).std().alias(f"{m}_sd") for m in ("MAE", "RMSE", "skill_vs_mta_%")])
            .sort("band", "MAE"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*")
    p.add_argument("--name", default="main")
    p.add_argument("--routes", nargs="*", help="score only these routes (e.g. the held-out ones)")
    args = p.parse_args()
    models = args.models or sorted(f.stem for f in PREDS.glob("*.parquet") if not f.stem.startswith("_"))
    df = load_common(models, args.routes)
    res = score(df, models)
    sig = significance(df, models)
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.write_csv(RESULTS / f"{args.name}.csv")
    if sig.height:
        sig.write_csv(RESULTS / f"{args.name}_significance.csv")
    seeds = seed_summary(res)
    seeds.write_csv(RESULTS / f"{args.name}_seeds.csv")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_float_precision(1)
    print(f"{df.height:,} common test rows over {df['service_date'].n_unique()} days")
    print(res.sort("band", "MAE"))
    if sig.height:
        print(f"\npaired comparison vs {REFERENCE} (daily MAE; negative dMAE = better than reference)")
        pl.Config.set_float_precision(4)
        print(sig.sort("band", "dMAE_vs_ref"))
    pl.Config.set_float_precision(2)
    print("\nmean (sd) over seeds")
    print(seeds)


if __name__ == "__main__":
    main()
