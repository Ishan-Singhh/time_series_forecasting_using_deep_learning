"""Run a per-day job across a process pool sized to the PBS allocation."""
import os
from datetime import date, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Callable

import polars as pl

from bus_eta.config import n_cpus


def day_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def write_atomic(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.rename(path)


def run_days(fn: Callable[[date], str], days: list[date], workers: int,
             initializer: Callable[[], None] | None = None) -> None:
    """Apply fn to each day in parallel, printing each returned status line.

    Parallelism is one process per day. Each worker's polars thread pool is capped so that
    workers x threads matches the allocated cores instead of oversubscribing them; spawned
    workers inherit the env var before polars initialises its pool.
    """
    workers = max(1, min(workers, len(days)))
    os.environ["POLARS_MAX_THREADS"] = str(max(1, n_cpus() // workers))
    with get_context("spawn").Pool(workers, initializer=initializer) as pool:
        for msg in pool.imap_unordered(fn, days):
            print(msg, flush=True)
