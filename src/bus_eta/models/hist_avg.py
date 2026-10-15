"""Historical-average baseline: the mean observed travel time, from departing stop k to
arriving at stop k+h, over TRAIN for the same route, direction, stop, horizon, day type
(weekday / Saturday / Sunday or holiday) and hour of day. No live information at all.

Cells with fewer than MIN_N training observations back off to coarser keys (all hours of
the day type, then all days), and finally to the scheduled running time.

    python -m bus_eta.models.hist_avg            # -> preds/hist_avg.parquet
"""
import polars as pl

from bus_eta.evaluate import KEY, PREDS
from bus_eta.features import EVAL_FRAC, load

MIN_N = 5
BASE = ["route_id", "direction_id", "stop_id", "h"]
LEVELS = [BASE + ["day_type", "hour"], BASE + ["day_type"], BASE]


def with_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("is_holiday") == 1).then(2).when(pl.col("dow") == 7).then(2)
        .when(pl.col("dow") == 6).then(1).otherwise(0).cast(pl.Int8).alias("day_type"),
        pl.col("tod_hour").floor().cast(pl.Int8).alias("hour"))


def main() -> None:
    cols = BASE + ["y", "tod_hour", "dow", "is_holiday"]
    train = with_keys(load("train").select(cols))
    print(f"train rows {train.height:,}", flush=True)
    test = with_keys(load("test", issue_frac=EVAL_FRAC).select(list(dict.fromkeys(KEY + cols + ["run_sched"]))))
    pred = pl.lit(None, dtype=pl.Float64)
    for i, keys in enumerate(LEVELS):
        cell = (train.group_by(keys).agg(pl.col("y").mean().alias(f"ha{i}"), pl.len().alias("n"))
                .filter(pl.col("n") >= MIN_N).drop("n"))
        test = test.join(cell, on=keys, how="left")
        pred = pl.coalesce(pred, pl.col(f"ha{i}"))
    test = test.with_columns(pl.coalesce(pred, pl.col("run_sched")).alias("pred"))
    used = [test[f"ha{i}"].is_not_null().mean() for i in range(len(LEVELS))]
    print(f"test rows {test.height:,}; cell found at level 0/1/2: " + " / ".join(f"{u:.1%}" for u in used), flush=True)
    test.select(KEY + ["pred"]).write_parquet(PREDS / "hist_avg.parquet")
    print("wrote preds/hist_avg.parquet")


if __name__ == "__main__":
    main()
