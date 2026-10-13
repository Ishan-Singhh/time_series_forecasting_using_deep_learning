"""Non-learned baselines on the test split, and the shared ground-truth file.

  sched    scheduled arrival
  rt       real-time profile (latest observed segment times of any bus)
  mta      MTA Bus Time's published prediction, as is
  mta_cal  MTA with a per-horizon linear correction (y ~ a_h + b_h * mta) fitted on TRAIN
           only, which removes MTA's arrival-vs-departure offset and horizon-dependent bias

    python -m bus_eta.models.baselines
"""
import numpy as np
import polars as pl

from bus_eta.config import H
from bus_eta.evaluate import KEY, PREDS
from bus_eta.features import EVAL_FRAC, load


def main() -> None:
    PREDS.mkdir(parents=True, exist_ok=True)
    test = load("test", issue_frac=EVAL_FRAC)
    test.select(KEY + ["y", "service_date", "route_id", "tod_hour", "dow"]).write_parquet(PREDS / "_truth.parquet")

    for name in ["sched", "rt", "mta"]:
        test.select(KEY + [pl.col(name).alias("pred")]).write_parquet(PREDS / f"{name}.parquet")

    train = load("train", issue_frac=0.05).drop_nulls("mta")
    coef = {}
    for h in range(1, H + 1):
        t = train.filter(pl.col("h") == h)
        b, a = np.polyfit(t["mta"].to_numpy(), t["y"].to_numpy(), 1)
        coef[h] = (a, b)
    cal = pl.DataFrame({"h": list(coef), "a": [c[0] for c in coef.values()], "b": [c[1] for c in coef.values()]},
                       schema={"h": pl.Int8, "a": pl.Float64, "b": pl.Float64})
    print(cal)
    (test.join(cal, on="h").select(KEY + [(pl.col("a") + pl.col("b") * pl.col("mta")).alias("pred")])
     .write_parquet(PREDS / "mta_cal.parquet"))
    print(f"wrote baselines for {test.height:,} test rows")


if __name__ == "__main__":
    main()
