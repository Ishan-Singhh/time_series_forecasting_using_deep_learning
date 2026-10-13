"""Re-implementation of the NYC next-stop model of Rashvand et al. (arXiv 2303.15495).

A fully connected network (320-200-100-40-5, ReLU, min-max scaled inputs, MSE loss, Adam
lr 1e-2 as in the paper) on their six inputs: one-hot route, distance to the next stop,
day type (weekday/weekend), rush hour (6-10 h, 15-19 h), integer-encoded stop and
"far" status (distance > 250 m). It predicts only the next stop (h = 1).

The paper's label was the feed's ArrivalTime, i.e. MTA's own expected arrival. With
--target mta we train on MTA's prediction instead of the GPS-observed arrival and report
the error against both, which measures how much that label choice flatters a model.

    python -m bus_eta.models.rashvand_mlp --name rashvand_mlp
    python -m bus_eta.models.rashvand_mlp --name rashvand_mlp_mtalabel --target mta
"""
import argparse

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from bus_eta.config import n_cpus
from bus_eta.evaluate import KEY, PREDS
from bus_eta.features import EVAL_FRAC, load

FAR_M = 250.0


def h1(split: str, frac: float, seed: int) -> pl.DataFrame:
    df = load(split, issue_frac=frac, seed=seed).filter(pl.col("h") == 1)
    hour = pl.col("tod_hour")
    return df.with_columns(
        pl.col("is_weekend").cast(pl.Float32).alias("day_type"),
        (hour.is_between(6, 10, closed="left") | hour.is_between(15, 19, closed="left")).cast(pl.Float32).alias("rush"),
        (pl.col("dist") > FAR_M).cast(pl.Float32).alias("far"),
    )


class Encoder:
    """Min-max scaling of the numeric inputs and one-hot routes, fitted on train."""

    def __init__(self, df: pl.DataFrame):
        self.routes = {r: i for i, r in enumerate(df["route_id"].unique().sort().to_list())}
        self.stops = {s: i + 1 for i, s in enumerate(df["stop_id"].unique().sort().to_list())}
        num = self.numeric(df)
        self.lo, self.hi = np.nanmin(num, 0), np.nanmax(num, 0)

    def numeric(self, df):
        stop = df["stop_id"].replace_strict(self.stops, default=0, return_dtype=pl.Float32).to_numpy()
        cols = [df[c].cast(pl.Float32).to_numpy() for c in ("dist", "day_type", "rush", "far")]
        return np.stack(cols[:1] + cols[1:3] + [stop] + cols[3:], 1)

    def __call__(self, df) -> torch.Tensor:
        num = np.nan_to_num((self.numeric(df) - self.lo) / np.maximum(self.hi - self.lo, 1e-6))
        onehot = np.zeros((df.height, len(self.routes)), dtype=np.float32)
        r = df["route_id"].replace_strict(self.routes, default=-1, return_dtype=pl.Int64).to_numpy()
        onehot[r >= 0, r[r >= 0]] = 1.0
        return torch.from_numpy(np.concatenate([onehot, num], 1).astype(np.float32))


def mlp(n_in: int) -> nn.Module:
    sizes = [n_in, 320, 200, 100, 40, 5]
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(5, 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--target", choices=["observed", "mta"], default="observed")
    p.add_argument("--train-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(n_cpus())
    label = "y" if args.target == "observed" else "mta"

    train = h1("train", args.train_frac, args.seed).drop_nulls(label)
    val = h1("val", 0.1, args.seed).drop_nulls(label)
    test = h1("test", EVAL_FRAC, 0)
    enc = Encoder(train)
    xtr, ytr = enc(train), torch.from_numpy(train[label].to_numpy().astype(np.float32))
    xva, yva = enc(val), val[label].to_numpy()
    print(f"{args.name}: train {len(ytr):,} next-stop rows, {xtr.shape[1]} inputs, label={label}", flush=True)

    model = mlp(xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=1)
    best, state, bad = np.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for idx in torch.randperm(len(ytr)).split(4096):
            loss = nn.functional.mse_loss(model(xtr[idx]).squeeze(-1), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(xva).squeeze(-1).numpy()
        rmse = float(np.sqrt(np.mean((pv - yva) ** 2)))
        sched.step(rmse)
        print(f"epoch {epoch:2d} val RMSE vs {label} {rmse:.2f}s  MAE {np.mean(np.abs(pv - yva)):.2f}s", flush=True)
        if rmse < best - 0.01:
            best, state, bad = rmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model(enc(test)).squeeze(-1).numpy()
    for name, truth in (("observed arrival", test["y"].to_numpy()), ("MTA prediction", test["mta"].to_numpy())):
        ok = np.isfinite(truth)
        e = pred[ok] - truth[ok]
        print(f"test vs {name}: RMSE {np.sqrt(np.mean(e ** 2)):.2f}s  MAE {np.mean(np.abs(e)):.2f}s  (n={ok.sum():,})")
    out = PREDS / "analysis"   # next-stop only: keep out of the main table (evaluate --models analysis/<name>)
    out.mkdir(parents=True, exist_ok=True)
    test.select(KEY).with_columns(pl.Series("pred", pred)).write_parquet(out / f"{args.name}.parquet")


if __name__ == "__main__":
    main()
