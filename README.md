# Multi-step bus arrival time prediction — Manhattan MTA network

**Task.** For a bus in service at time *t*, predict its arrival time at each of the next
*H* downstream stops (horizons of ~1–30 minutes). The models are evaluated against the
predictions MTA Bus Time actually published at the same instant, which are archived
in the trip-updates feed.

**Why this formulation.** An earlier version of this project predicted single
stop-to-stop segment times. That target is mostly irreducible noise (traffic signals,
dwell). Even an oracle that knew other buses' times on the same segment in the same
15–30 minute window reached only R² ≈ 0.2–0.3. Over a 10-stop horizon the same oracle
reached R² ≈ 0.82. Multi-step ETA is also the quantity riders and agencies actually use.

## Data

| Source | Coverage |
|---|---|
| GTFS-RT vehicle positions + trip updates (`parquet.gtfsrt.io` archive of MTA Bus Time) | 2026-03-01 → present, all 42 Manhattan routes |
| Static GTFS (Manhattan), one per schedule pick: A6, B6, C6, D6 | trip_id → shape, stop pattern, schedule |
| Open-Meteo hourly weather, calendar/holidays, major events | context features |

Ground-truth arrival times are **observed**, not taken from MTA predictions. They are derived
by projecting each GPS ping onto the trip's shape and interpolating the time the bus crossed
each stop. (About 21% of segment labels built from trip updates were really MTA predictions.)

## Pipeline

```
src/bus_eta/
  config.py          paths, split dates, horizon H, n_cpus() (reads PBS NCPUS)
  parallel.py        per-day process pool sized to the allocation
  fetch_rt.py        daily RT download (resumable, change-only trip updates)      [login node]
  fetch_static.py    dated snapshot of the current static GTFS                     [login node]
  context.py         hourly weather (Open-Meteo) + daily calendar                  [login node]
  geo.py             projection + numba kernels (stops/GPS -> distance along shape)
  static_gtfs.py     per-pick trips, stop patterns, stop distances, service dates
  arrivals.py        observed stop arrival/departure times from GPS (ground truth)
  samples.py         one row per observed departure: targets for the next H stops,
                     MTA prediction in force, real-time profile ETA, bus state
  features.py        long (issue x horizon) modelling table, split loader
  evaluate.py        common-support scoring by horizon band, day-bootstrap CIs
  models/
    baselines.py     schedule, real-time profile, MTA, train-calibrated MTA
    lgbm.py          LightGBM multi-horizon (with / without MTA as an input)
jobs/                PBS scripts: cpu.pbs (any module), lgbm.pbs
```

Order: `fetch_rt`, `fetch_static`, `context` (login node) → `static_gtfs` → `arrivals` →
`samples` (CPU node) → `models.baselines`, `models.lgbm` → `evaluate`.

**Ground-truth notes.** GPS-derived arrivals cover ~93% of each trip's stops; 96% are
bracketed by pings ≤ 60 s apart. MTA's own `arrival_time` matches the bus's GPS
*departure* (83% within 30 s), so the raw MTA baseline carries a ~20 s offset; `mta_cal`
removes it using training data only.

**Split (chronological, fixed):** train Mar–Jul, validation Aug, test Sep–Oct.
The test set is never used for early stopping or tuning. Every result is reported as mean ± std over seeds.

## Setup

```bash
source /apps/anaconda3/bin/activate bus_eta
pip install -e .

# Downloads run on the LOGIN node: compute nodes have no internet access.
nohup python -m bus_eta.fetch_rt --start 2026-03-01 --end 2026-10-12 > jobs/logs/fetch_login.log 2>&1 &
python -m bus_eta.fetch_static     # re-run after each new schedule pick
```

GPU work (training) runs as PBS jobs on the `gpu` queue, since the login node has no GPU.
