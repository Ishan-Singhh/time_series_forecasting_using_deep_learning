"""LightGBM multi-horizon ETA: one model over all horizons (h is a feature).

Target is the correction to the real-time profile ETA (y - rt), so trees model a
residual of similar scale at every horizon; prediction = rt + f(x). Early stopping
uses the validation month only.

    python -m bus_eta.models.lgbm --name lgbm_no_mta
    python -m bus_eta.models.lgbm --name lgbm_mta --use-mta
    python -m bus_eta.models.lgbm --name lgbm_no_mta_no_stop --drop stop_id     # ablation
    python -m bus_eta.models.lgbm --name lgbm_mta_randsplit --use-mta --split random  # leakage pitfall
    python -m bus_eta.models.lgbm --name lgbm_mta_randsplit --predict-only --drop-boundary  # rescore saved model
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from bus_eta.config import PROCESSED, SPLITS, n_cpus
from bus_eta.evaluate import KEY, PREDS
from bus_eta.features import CATEGORICAL, EVAL_FRAC, feature_columns, load

MODELS = PROCESSED / "models"


def encode(df: pl.DataFrame, cols: list[str], cat_maps: dict) -> np.ndarray:
    exprs = []
    for c in cols:
        if c in CATEGORICAL:
            # Unseen categories (e.g. new stops in a later schedule pick) become missing.
            exprs.append(pl.col(c).replace_strict(cat_maps[c], default=None, return_dtype=pl.Float32))
        else:
            exprs.append(pl.col(c).cast(pl.Float32))
    return df.select(exprs).to_numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--use-mta", action="store_true")
    p.add_argument("--train-frac", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=20000, help="cap; early stopping on val decides")
    p.add_argument("--drop", nargs="*", default=[], help="features to remove (ablation)")
    p.add_argument("--school", action="store_true", help="add the NYC public-school day flag")
    p.add_argument("--attrs", action="store_true", help="add path signals / bus-lane attributes")
    p.add_argument("--holdout-routes", nargs="*", default=[], help="routes removed from train/val (cold start)")
    p.add_argument("--split", choices=["chrono", "random"], default="chrono",
                   help="random = issues of all months shuffled into train/val (pitfall experiment)")
    p.add_argument("--predict-only", action="store_true", help="score the saved booster of --name, no training")
    p.add_argument("--drop-boundary", action="store_true",
                   help="leave test issues whose service_date is before the test range (late trips of Aug 31 "
                        "in the Sep 1 file) out of scoring; random-split models trained before that fix saw them")
    args = p.parse_args()
    if args.predict_only:
        meta = json.loads((MODELS / f"{args.name}.json").read_text())
        saved = meta["args"]
        booster = lgb.Booster(model_file=str(MODELS / f"{args.name}.txt"))
        test = load("test", issue_frac=EVAL_FRAC, attrs=saved["attrs"])
        if args.drop_boundary:
            n = test.height
            test = test.filter(pl.col("service_date") >= SPLITS["test"][0])
            print(f"dropped {n - test.height:,} boundary test rows", flush=True)
        predict(booster, test, meta["features"], meta["cat_maps"], saved)
        return

    cols = [c for c in feature_columns(args.use_mta, args.school, args.attrs) if c not in args.drop]
    cats = [c for c in CATEGORICAL if c in cols]
    tr, va = ("train", "val") if args.split == "chrono" else ("random_train", "random_val")
    train = load(tr, issue_frac=args.train_frac, seed=args.seed, attrs=args.attrs)
    val = load(va, issue_frac=args.val_frac if args.split == "chrono" else 1.0, seed=args.seed, attrs=args.attrs)
    if args.holdout_routes:  # unseen route_id / stop_id become missing at test (replace_strict default)
        train = train.filter(~pl.col("route_id").is_in(args.holdout_routes))
        val = val.filter(~pl.col("route_id").is_in(args.holdout_routes))
    print(f"train {train.height:,} rows, val {val.height:,} rows, {len(cols)} features", flush=True)

    cat_maps = {c: {v: i for i, v in enumerate(train[c].unique().sort().to_list())} for c in cats}
    cat_idx = [cols.index(c) for c in cats]
    dtrain = lgb.Dataset(encode(train, cols, cat_maps), (train["y"] - train["rt"]).to_numpy(),
                         feature_name=cols, categorical_feature=cat_idx, free_raw_data=True)
    dval = lgb.Dataset(encode(val, cols, cat_maps), (val["y"] - val["rt"]).to_numpy(), reference=dtrain)
    issue = ["trip_id", "start_date", "k"]
    train_issues = train.select(issue).unique()
    del train

    params = {
        "objective": "l1", "metric": "l1", "learning_rate": 0.05, "num_leaves": 255,
        "min_data_in_leaf": 500, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "cat_smooth": 50, "max_cat_to_onehot": 8, "num_threads": n_cpus(), "seed": args.seed,
        "verbosity": -1,
    }
    booster = lgb.train(params, dtrain, num_boost_round=args.rounds, valid_sets=[dval], valid_names=["val"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

    MODELS.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODELS / f"{args.name}.txt"))
    (MODELS / f"{args.name}.json").write_text(json.dumps({"features": cols, "cat_maps": cat_maps,
                                                          "args": vars(args)}))

    test = load("test", issue_frac=EVAL_FRAC, attrs=args.attrs)
    overlap = test.select(issue).unique().join(train_issues, on=issue).height
    assert overlap == 0, f"{overlap} scored test issues were also trained on"
    predict(booster, test, cols, cat_maps, vars(args))


def predict(booster: lgb.Booster, test: pl.DataFrame, cols: list[str], cat_maps: dict, args: dict) -> None:
    pred = test["rt"].to_numpy() + booster.predict(encode(test, cols, cat_maps), num_threads=n_cpus())
    side = args["split"] != "chrono" or args["holdout_routes"]  # side analyses
    out = PREDS / "analysis" if side else PREDS
    out.mkdir(parents=True, exist_ok=True)
    test.select(KEY).with_columns(pl.Series("pred", pred)).write_parquet(out / f"{args['name']}.parquet")

    imp = sorted(zip(cols, booster.feature_importance("gain")), key=lambda t: -t[1])
    total = sum(g for _, g in imp)
    print("top features by gain:", ", ".join(f"{c} {100 * g / total:.1f}%" for c, g in imp[:12]))


if __name__ == "__main__":
    main()
