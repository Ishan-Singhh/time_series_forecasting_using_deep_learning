# Research log: multi-stop bus arrival prediction (Manhattan, MTA)

Living record of the setup, decisions, results and open items, written for drafting the
paper. Numbers are on the test period unless stated. "MAE" means mean absolute error in
seconds of the predicted arrival time. Update this file whenever a result comes in.

Last updated: 2026-10-14 (branch `restart-multistep-eta`)

---

## 1. Problem and framing

- **Task.** When a bus departs stop k at time t, predict its arrival time at each of the
  next H = 20 stops (about 30 min ahead), using only information available at t.
- **Why this task, not single-segment travel time (the old project).** The single-segment
  target is dominated by noise. Every old model plateaued at MAE ≈ 29 s and R² ≈ 0.5. An
  oracle using other buses on the same segment and time window only reached R² 0.21–0.31.
  Over a 10-stop horizon the same oracle reached R² 0.82, so multi-stop targets carry
  learnable signal.
- **Headline benchmark.** MTA Bus Time's own published predictions, i.e. the operator's
  deployed system, archived from the GTFS-RT TripUpdates feed. Also its calibrated version
  (`mta_cal`, a per-horizon linear correction fitted on training data), which is the
  reference for all skill and significance numbers.
- **Proposed contribution (current form).**
  1. A citywide, open-data, multi-stop ETA setup with GPS ground truth and a strict
     chronological evaluation.
  2. A corridor graph whose segments are shared across routes, so all buses act as
     traffic probes for each other.
  3. A model (graph encoder + trip transformer) that beats MTA by 16% MAE, with gains
     growing with horizon.
  4. Extensive ablations, robustness and evaluation-pitfall analyses.

## 2. Data

| Item | Detail |
|---|---|
| Region | All 42 Manhattan "M*" routes (incl. SBS "+"); 41 routes appear in test |
| Real-time feeds | parquet.gtfsrt.io archive of MTA Bus Time GTFS-RT, 2026-03-01 → 2026-10-12 (225 service days). Vehicle positions 1.4 GB; trip updates stored change-only, 7.5 GB; service alerts |
| Missing upstream | Trip updates for 2026-09-05. VP odometer, speed, current_stop_sequence and status fields are null |
| Static GTFS | 4 schedule picks A6/B6/C6/D6 (gtfs_m.zip). Trip-id coverage ≥ 92% every day (May 23–25 < 95%) |
| Weather | Open-Meteo archive, Central Park, hourly |
| Calendar | US federal holidays; major Manhattan events CSV; **NYC public-school days from the official NYCPS 2025-26 and 2026-27 calendars** (`nyc_school_calendar.csv`; 93 school days in the study period; school starts Sep 10, inside the test window) |
| Signals | OpenStreetMap `highway=traffic_signals`, 7,763 nodes in the Manhattan bounding box, merged within 25 m into intersections |
| Bus lanes | NYC DOT "Bus Lanes – Local Streets" (ycrg-ses3): 1,304 Manhattan segments, about 90 km; 799 dedicated, 353 shared. 736 are 24/7; the rest are part-time (mostly weekdays 7–10 AM / 2–7 PM) |
| Service alerts | MTA bus alerts Mar 1–Oct 12. Route-level free text only (cause/effect empty), about 9 Manhattan alerts per day. Used only in the error analysis (§5.6); not a model input |
| Rejected: traffic speeds | NYC DOT real-time speeds (i4gi-tjb9) cover only **26 Manhattan links, all highways, bridges and tunnels** (FDR, West Side Hwy, Lincoln/BBT/QMT/GWB), about half invalid (status -101). No coverage of the avenues buses use. Commercial feeds aren't open; Uber Movement was discontinued. Paper point: the bus fleet itself is the street-level traffic sensor network |

### Ground truth (paper section: Data)
- Arrivals and departures are reconstructed from **GPS positions projected onto route
  shapes**, not from MTA predictions.
  - Arrival = 10 m before the stop; departure = 15 m after it.
  - About 93% of stops are labelled; 96% of labels come from GPS pings ≤ 60 s apart.
- **Finding: MTA's TripUpdate "arrival_time" is really a departure time.** It matches the
  GPS departure within 30 s for 83% of stops, about a 20 s offset from arrival. This
  motivates `mta_cal`.
- **Finding: label contamination in the old work.** About 21% of the old segment labels
  were MTA predictions, not observations.

### Samples
- One sample ("issue") per observed departure. Issue time = when the departure became
  visible in the feed (`dep_seen`). Each issue predicts the next 20 stops.
- 225 daily files, about 210k issues per day.
- The neural packs subsample issues by a deterministic hash:
  - train: 5.82M issues, 75.5M targets;
  - val: 602k issues, 7.9M targets;
  - test: 2.00M issues, 26.0M targets.
- Two vehicles occasionally report the same trip_id (about 0.02% of issues). Those keys
  are dropped as ambiguous. Unchecked, they had multiplied across model joins in an early
  evaluation (bug found and fixed).

### Data rebuild (2026-10-16, branch `fix-utc-day-boundary`)
Two data bugs were found and fixed. Everything downstream was rebuilt (27396) and every model
is being retrained. The old build is kept in `data/archive_utc/`.
1. **UTC day files split trips (§7).**
   - Arrivals are now keyed by service date. Day d is built from the archive files of d and
     d+1 and keeps the trips of d, so every trip is projected once, from its whole track.
   - Samples take other buses' runs, dwells and headways from days d−1..d+1. These are as-of
     joins on reveal time, so they stay causal. MTA predictions come from files d and d+1.
   - Pilot, Sep 14–18:
     - 0 trips in two files (was 1,178);
     - targets per issue at 19:45–20:00 rose to 12.86 (was 4.74);
     - +0.85% targets;
     - targets already present were 99.99% identical.
   - **The old build also lost most traversals of split trips.** Each half kept a row for
     every stop of the trip: rows with NaN times passed the not-null filter, because NaN was
     only turned into null after filtering. In `network_state`, and in `samples`' real-time
     history, the two copies of each stop interleaved, which broke the consecutive-stop
     test.
     - On Sep 16, the 299 split trips gave 2,624 traversals, against 10,752 after the fix.
     - Over the study period, traversals went from 40.83M to 42.70M (+1.87M, +4.6%).
       Almost all of the gain is at 18–21 h local: hour 19 went from 1.38M to 2.25M and
       hour 20 from 1.20M to 2.02M. Every other hour changed by < 0.1%.
     - Stop rows fell from 54.8M to 52.1M, because the duplicate NaN rows are gone.
     - So the old network state was about 40% blind at 19–21 h, when the boundary is
       00:00 UTC.
2. **The feed's start_date was wrong on the DST change.**
   - On 2026-03-08, MTA published every Sunday trip (5,782 trips) with start_date
     20260307. The day had no file of its own; its trips were filed under Mar 7 with
     schedules 24 h off.
   - The service date is now taken from a trip's first ping and its scheduled first
     departure. start_date stays the join key with TripUpdates.
   - Scheduled times are now counted from GTFS's "noon − 12 h", not from local midnight.
     The two differ by an hour on DST days: Mar 8's median schedule delay was −3,522 s and
     is now 78 s.

3. **Consequence: the test period is 41 service days (Sep 1 – Oct 11), not 42.**
   - The old build's "42 days" counted 123k test rows of Aug 31 trips that ran past 00:00
     UTC into the Sep 1 file. Their earlier stops sat in the Aug 31 file, i.e. in
     validation, so a few trip instances straddled the val/test boundary. That is now gone.
   - Oct 11 also gains its evening: 530k rows, up from 407k.
   - Every daily paired test now uses 41 days.
4. **Early check (2026-10-16, `results/early.csv`; the models finished so far, on their
   own common support of 25.61M rows).** Compared with the old final table:
   - **Ours** (`stgnn_mta_qsdrop`, 3 seeds): 97.03 s, was 96.72; skill vs `mta_cal`
     **+16.4%**, unchanged.
   - **LightGBM** (`lgbm_mta`): 99.06 s, was 99.20; RMSE **163.3, was 183.3**. The old
     LightGBM's heavy tail came largely from the broken evening hours.
   - **MTA:** `mta_cal` 116.09 s, was 115.68.
   - Absolute MAEs rise slightly because the recovered evening rows include more long
     horizons.
   - Daily paired vs `mta_cal`: ours −19.1 s (100% of 41 days); LightGBM −17.0 s.

Rebuilt sizes:
- network state: 42.7M traversals, 26.0% of node-bins observed (was 25%);
- packs:
  - train 5.83M issues / 76.35M targets (was 75.5M);
  - val 603k / 7.97M;
  - test 2.00M / **26.29M** (was 26.0M).

### Splits and protocol (paper section: Experimental setup)
- **Chronological split:**
  - train 2026-03-01 → 07-31;
  - validation 08-01 → 08-31;
  - test 09-01 → 10-11 (41 service days; a new schedule pick D6).
- Early stopping and model selection use **validation only**; the test set is never used
  for tuning.
- **Scoring:**
  - A fixed 25% hash sample of test issues (`EVAL_FRAC = 0.25`), the same for every model.
    The final paper tables should use 100%.
  - Every model is scored on the **common support**: rows where all compared models
    predict.
- **Metrics:** MAE with day-bootstrap 95% CI, MdAE, RMSE, MAPE (y ≥ 60 s), WAPE, R², P90
  absolute error, median bias, % within 60/120 s, skill vs `mta_cal`.
- **Horizon bands:** h1, h2–5, h6–10, h11–15, h16–20, all.
- **Significance:** paired comparison on **daily MAE** (days are the independent unit):
  bootstrap CI of the difference and a Wilcoxon signed-rank test.
- **Seeds:** 3 per neural model, reported as mean ± sd.

## 3. Models

| Name | Description |
|---|---|
| `sched` | Static schedule |
| `rt` | Real-time profile: latest observed run and dwell time on each downstream segment by any bus (≤ 30 min old), summed |
| `mta` / `mta_cal` | MTA's published prediction / per-horizon linear calibration |
| `lgbm_*` | LightGBM, one model for all horizons (h is a feature). Target = y − rt; L1 loss; early stop on val. 20% of training issues (about 75M rows), 30–32 features |
| `seq_*` | Neural model without the graph: issue token + 20 downstream-stop tokens → Transformer (3 layers, d = 128) → residual over rt; masked L1 |
| `stgnn_*` | Same plus a **graph encoder** over the corridor network state, read at each token's segment |
| `gwnet_*`, `dcrnn_*` | Same decoder with a faithful Graph WaveNet (Wu et al. 2019) or DCRNN (Li et al. 2018) encoder |
| `rashvand_mlp` | Re-implementation of Rashvand et al.'s NYC next-stop MLP (6 inputs, 320-200-100-40-5, MSE) |
| `_mta` / `_no_mta` | With or without MTA's prediction as an input |

**Corridor graph.**
- 2,111 directed stop-to-stop segment nodes shared by every route that uses them; 545 are
  shared by more than one route.
- Edges: 2,220 flow edges (consecutive segments) and 1,671 corridor edges (segments within
  150 m with heading within 30°, e.g. local vs SBS on the same street).

**Network state.**
- 5-min bins × node × (speed, #traversals, dwell, observed-mask).
- Every traversal is filed under the bin in which it was **revealed**, so it is causal.
- 40.8M traversals; 25% of node-bins observed.
- The encoder reads the last 12 completed bins.

**Our encoder.** Per-node GRU over the 12 bins, then 2 graph layers mixing flow-forward,
flow-backward (spillback) and corridor neighbours plus a learned adaptive adjacency.

**Training.**
- AdamW with a one-cycle learning rate up to 1e-3, 10 epochs, patience 3.
- Batches are grouped by time bin, so the graph is encoded once per bin.
- bf16 autocast; graph operations in fp32.

**Compute.**
- Shared 2× NVIDIA L40 on a PBS cluster; 4 CPUs / 28 GB per GPU job.
- Epoch times: no-graph about 2–4 min; ours 7–15 min; Graph WaveNet about 23 min; DCRNN
  about 35–110 min, since its sequential diffusion steps make it launch-bound.

## 4. Main results (test; `all` = horizons 1–20; mean ± sd over seeds)

Rebuilt data (2026-10-17): 41 test days (Sep 1–Oct 11), 25,608,377 rows on the common
support of every model below. Tables: `results/final*.csv` (`python -m bus_eta.evaluate
--name final`) and `results/main*.csv` (MTA-input and no-graph ablations, `--name main`).
Every neural run uses one setting: 10 epochs, patience 3, batch 2048, max-bins 16.

| Model | Seeds | MAE (s) | RMSE | MAPE % | ≤ 60 s % | Skill vs mta_cal |
|---|---|---|---|---|---|---|
| **stgnn_mta_q (main model: quantile head)** | 3 | **96.61 ± 0.11** | 155.8 | 15.1 | 49.3 | **+16.8%** |
| stgnn_mta (original, L1) | 3 | 96.80 ± 0.19 | 156.5 | 15.1 | 49.3 | +16.6% |
| gwnet_mta (Wu et al. 2019) | 3 | 96.84 ± 0.06 | 155.8 | 15.1 | 49.2 | +16.6% |
| gwnet_mta_q (GWNet + our quantile head) | 3 | 96.95 ± 0.32 | 156.0 | 15.1 | 49.2 | +16.5% |
| stgnn_mta_sdrop (obs. dropout) | 3 | 97.01 ± 0.20 | 156.7 | 15.1 | 49.2 | +16.4% |
| stgnn_mta_qsdrop (robust variant: + obs. dropout) | 3 | 97.06 ± 0.20 | 157.0 | 15.1 | 49.2 | +16.4% |
| stgnn_no_mta | 3 | 97.07 ± 0.13 | 156.7 | 15.2 | 49.1 | +16.4% |
| dcrnn_mta (Li et al. 2018) | 3 | 97.59 ± 0.14 | 158.2 | 15.2 | 49.0 | +15.9% |
| a3tgcn_mta (Bai et al. 2021) | 3 | 97.85 ± 0.10 | 159.0 | 15.3 | 49.0 | +15.7% |
| tgcn_mta (Zhao et al. 2020) | 1 | 97.87 | 159.4 | 15.2 | 49.1 | +15.7% |
| seq_mta (no graph) | 3 | 98.82 ± 0.12 | 161.3 | 15.3 | 48.8 | +14.9% |
| staeformer_mta (Liu et al. 2023) | 3 | 98.98 ± 0.21 | 161.0 | 15.4 | 48.7 | +14.7% |
| lgbm_mta (20% of train issues) | 1 | 99.06 | 163.3 | 15.4 | 48.7 | +14.7% |
| convlstm_mta_e20 (Petersen et al. 2019, 20 ep) | 3 | 99.41 ± 0.08 | 161.8 | 15.4 | 48.6 | +14.4% |
| seq_no_mta | 3 | 100.17 ± 0.23 | 164.5 | 15.5 | 48.4 | +13.7% |
| lgbm_no_mta | 1 | 100.54 | 166.3 | 15.6 | 48.3 | +13.4% |
| hist_avg (historical average, no live data) | – | 115.34 | 190.8 | 17.9 | 44.0 | +0.6% |
| mta_cal | – | 116.09 | 192.2 | 18.4 | 43.3 | 0 |
| mta (published) | – | 117.85 | 198.1 | 18.7 | 43.2 | −1.5% |
| rt | – | 134.30 | 210.0 | 21.4 | 38.0 | −15.7% |
| sched | – | 356.18 | 590.9 | 81.6 | 17.1 | −206.8% |

- 95% day-bootstrap CI of the MAE: 94.1–99.6 for stgnn_mta_q (seed 0; qsdrop 94.5–100.0),
  113.1–119.9 for `mta_cal`. Median bias −3.5 s (ours) vs +7.4 s (`mta_cal`). Within
  120 s: 73.6% vs 67.8%.
- Seed-averaged paired tests (daily MAE averaged over seeds, 41 days, Wilcoxon;
  `python -m bus_eta.paired`, rerun 2026-10-18 with the main model first). Daily means
  differ from the pooled table by < 0.15 s.

| Comparison | ΔMAE (s) | Days better | p |
|---|---|---|---|
| **Main model (stgnn_mta_q) vs …** | | | |
| stgnn_mta (original, L1) | −0.22 | 85% | 1e-8 |
| stgnn_mta_qsdrop (robust variant) | −0.46 | 95% | 3e-11 |
| **gwnet_mta** | **−0.25** | 78% | 8e-5 |
| **gwnet_mta_q (GWNet + the same quantile head)** | **−0.34** | 80% | 8e-6 |
| dcrnn_mta | −1.01 | 100% | 9e-13 |
| a3tgcn_mta | −1.31 | 100% | 9e-13 |
| tgcn_mta | −1.34 | 100% | 9e-13 |
| seq_mta (no graph) | −2.35 | 100% | 9e-13 |
| staeformer_mta | −2.42 | 100% | 9e-13 |
| lgbm_mta | −2.61 | 100% | 9e-13 |
| convlstm_mta_e20 | −2.99 | 100% | 9e-13 |
| hist_avg | −18.59 | 100% | 9e-13 |
| **Robust variant (stgnn_mta_qsdrop) vs …** | | | |
| stgnn_mta (original) | +0.25 | 15% | 4e-6 |
| stgnn_mta_sdrop | +0.01 | 44% | 0.4 (tie) |
| gwnet_mta | +0.21 | 29% | 0.007 |
| gwnet_mta_q | +0.12 | 46% | 0.3 (tie) |
| dcrnn_mta | −0.55 | 98% | 4e-11 |
| seq_mta | −1.89 | 100% | 9e-13 |
| lgbm_mta | −2.15 | 100% | 9e-13 |
| **Ablations and baselines** | | | |
| stgnn_mta_sdrop vs stgnn_mta | +0.24 | 17% | 8e-8 |
| stgnn_mta vs gwnet_mta | −0.03 | 56% | 0.2 (tie) |
| stgnn_mta vs dcrnn_mta | −0.80 | 100% | 9e-13 |
| gwnet_mta vs dcrnn_mta | −0.77 | 95% | 6e-12 |
| gwnet_mta_q vs gwnet_mta (quantile head on GWNet) | +0.09 | 34% | 0.007 |
| gwnet_mta_q vs stgnn_mta | +0.13 | 29% | 0.01 |
| stgnn_mta vs seq_mta (graph gain, with MTA input) | −2.14 | 100% | 9e-13 |
| stgnn_no_mta vs seq_no_mta (graph gain, no MTA input) | −3.20 | 100% | 9e-13 |
| gwnet_mta vs seq_mta | −2.10 | 100% | 9e-13 |
| stgnn_mta vs stgnn_no_mta (MTA input, with graph) | −0.11 | 90% | 1e-6 |
| seq_mta vs seq_no_mta (MTA input, no graph) | −1.18 | 95% | 1e-8 |
| lgbm_mta vs lgbm_no_mta | −1.34 | 95% | 1e-8 |
| staeformer vs seq_mta | +0.07 | 34% | 0.2 (tie) |
| staeformer vs lgbm_mta | −0.20 | 46% | 1 (tie) |
| convlstm_mta_e20 vs lgbm_mta | +0.37 | 32% | 0.05 |

**Our encoder vs Graph WaveNet, by horizon band** (seed-averaged ΔMAE vs gwnet_mta, s;
days better):

| Model | h1 | h2–5 | h6–10 | h11–15 | h16–20 |
|---|---|---|---|---|---|
| stgnn_mta | −0.09 (93%) | −0.17 (88%) | −0.14 (76%) | +0.03 (54%, n.s.) | +0.32 (39%, p = 0.02) |
| stgnn_mta_q | −0.11 (93%) | −0.27 (95%) | −0.34 (85%) | −0.23 (76%) | −0.11 (66%, n.s.) |
| stgnn_mta_qsdrop | −0.07 (90%) | −0.06 (76%) | +0.10 (n.s.) | +0.41 (27%) | +0.76 (22%) |

**MAE by horizon band (s, mean over seeds):**

| Model | h1 | h2–5 | h6–10 | h11–15 | h16–20 |
|---|---|---|---|---|---|
| stgnn_mta_q (main) | 28.5 | **56.9** | **95.2** | **126.7** | **153.0** |
| stgnn_mta (original) | 28.6 | 57.0 | 95.4 | 127.0 | 153.4 |
| stgnn_mta_qsdrop (robust) | 28.6 | 57.1 | 95.7 | 127.4 | 153.8 |
| gwnet_mta | 28.6 | 57.2 | 95.6 | 127.0 | 153.1 |
| gwnet_mta_q | 28.7 | 57.2 | 95.6 | 127.1 | 153.4 |
| dcrnn_mta | 28.8 | 57.4 | 96.2 | 128.1 | 154.7 |
| a3tgcn_mta | 28.8 | 57.5 | 96.4 | 128.4 | 155.1 |
| tgcn_mta | 28.8 | 57.5 | 96.4 | 128.5 | 155.3 |
| seq_mta | 29.0 | 58.0 | 97.4 | 129.8 | 156.8 |
| staeformer_mta | 29.3 | 58.1 | 97.5 | 130.0 | 157.0 |
| convlstm_mta_e20 | 28.9 | 58.1 | 97.7 | 130.7 | 158.4 |
| lgbm_mta | **28.4** | 57.8 | 97.6 | 130.4 | 157.6 |
| hist_avg | 33.8 | 67.4 | 113.4 | 151.7 | 183.5 |
| mta_cal | 38.7 | 70.3 | 113.3 | 150.7 | 182.9 |

### Findings
1. **All learned models beat MTA's deployed predictions on all 41 test days**
   (p = 9e-13), in every horizon band. The main model (stgnn_mta_q) is +16.8% overall,
   +26.2% at h1 and +16.3% at h16–20; the historical average (+0.6%) ties MTA.
2. **The graph helps, and the help grows with horizon.**
   - Graph vs no graph (paired, daily): −2.14 s with MTA input, −3.20 s without; better
     on 100% of days.
   - By band (stgnn_mta − seq_mta): −0.5 s at h1, −3.5 s at h16–20 (no MTA input:
     −0.7 / −5.3 s).
   - Seed spread is 0.1–0.2 s, so the gain is 10–20× the noise.
   - Interpretation: congestion further downstream is invisible to the bus itself but
     observed by other buses already on that street.
3. **With the graph, MTA's predictions add almost nothing as input** (−0.11 s, 90% of
   days). An open-data model reconstructs and exceeds the operator's information. Without
   the graph, MTA input is worth 1.2 s (LightGBM: 1.3 s).
4. **The graph matters more than the encoder design — and on the rebuilt data our
   encoder no longer beats Graph WaveNet overall.**
   - stgnn_mta ties GWNet (−0.03 s, p = 0.2). It is better up to h10 and worse at
     h16–20 (+0.32 s). With the quantile head (the main model) it beats GWNet by 0.25 s
     (78% of days, p = 8e-5), in every band except h16–20 (n.s.).
   - **The quantile loss does not explain the win on its own** (2026-10-18). GWNet with the
     same quantile head (`gwnet_mta_q`, same 7 quantiles, 3 seeds, job 27485) gets slightly
     *worse*: 96.95 ± 0.32 s, +0.09 s vs GWNet with L1 (better on 34% of days, p = 0.007),
     and the main model beats it by 0.34 s (80% of days, p = 8e-6). Our encoder gains
     0.22 s from the pinball loss; GWNet loses 0.09 s. So the edge is the combination of our
     encoder and the quantile loss, and it survives giving the baseline the same loss.
     Still small (0.25–0.34 s), so the paper should say "matches or slightly beats GWNet",
     not "beats GWNet by a wide margin". GWNet's intervals with the quantile head are as
     well calibrated as ours (90%: 88.8% vs 89.0% coverage, width 377 vs 378 s), so the
     calibration comes from the loss, not the encoder.
   - The robust variant (qsdrop) is 0.21 s behind GWNet (p = 0.007), entirely at long
     horizons, and ties GWNet with the quantile head (+0.12 s, p = 0.3).
   - Old build: ours beat GWNet by 0.41 s (86% of days) with the gap growing with
     horizon. The old GWNet runs were trained on the broken evening data, and maybe
     with other settings (the old runs had mixed batch / max-bins), so this comparison is
     the one to trust. **Do not claim an encoder win over GWNet.**
   - Both typed-graph encoders (ours, GWNet) beat DCRNN by about 0.8 s (95–100% of days),
     and DCRNN beats A3T-GCN / T-GCN; all adjacency-based encoders beat no graph.
   - **STAEformer** (no adjacency; attention over all 2,111 segments) ties the no-graph
     model (+0.07 s, p = 0.2) and LightGBM (p = 1). It overfits: training MAE keeps falling
     (to 84 s) while validation stops at 89.0–89.2 s and early stopping fires (seeds 1–2).
     About 19 min/epoch alone, the most expensive baseline. Reading: on a sparse probe
     network, attention without the road graph recovers nothing over the bus's own
     trajectory; the physical adjacency carries the gain.
   - A3T-GCN ties T-GCN (97.85 vs 97.87): its attention adds nothing.
5. **Simple trees are as good one stop ahead.** LightGBM is the best model at h1 (28.4 s
   vs 28.5–28.6). Complexity pays off only at longer horizons.
6. **RMSE: neural models still fix the tail, but less than before.** On the rebuilt data
   LightGBM's RMSE is 163.3 (old 183.3; much of its old tail came from the broken
   evening data), vs 155.8–157.0 for the graph models and 192.2 for `mta_cal`.
7. **LightGBM details (old build, not rerun).** 5% → 20% of training issues improved MAE
   by about 2 s; dropping `stop_id` cost only 0.2 s.
8. **Training longer (20 epochs) didn't help on test (old build, not rerun):** 96.70 vs
   96.61 s, despite 0.3 s better validation. 10 epochs is the protocol.
9. **Consistency (old build):** predicted arrival times decrease from one stop to the next
   on only 0.19% of consecutive pairs for ours (MTA: 0.36%).
10. **Petersen et al.'s ConvLSTM** (`--arch convlstm`, re-implemented with the same inputs;
    a ConvLSTM over the (time × downstream link) grid of the bus's own route, all horizons
    at once), 20 epochs, 3 seeds: 99.41 s, +14.4% vs MTA (100% of days). It reproduces the
    paper's claim of beating the operator, but is behind no graph (+0.6 s) and level with
    LightGBM on MAE (+0.37 s, p = 0.05) with better RMSE (161.8 vs 163.3). The main model
    beats it by 3.0 s on 100% of days; route-local convolution sees only the bus's own
    corridor, the graph sees crossing and parallel routes.
11. **A historical average with no live data matches MTA's deployed predictions**
    (`python -m bus_eta.models.hist_avg`: mean TRAIN travel time from stop k to k+h for
    the same route, direction, stop, horizon, day type and hour, with back-off).
    115.34 s vs 116.09 for `mta_cal` (−0.9 s, p = 0.03 in the early check) and 117.85 for
    MTA as published; +12.7% at h1, level with MTA from h6 on. The operator's real-time
    system adds nothing over a lookup table of past travel times; every learned model beats
    it by 14.8–18.7 s. The open feed carries information MTA's own predictor doesn't use.
12. **Main model: `stgnn_mta_q`; robust variant: `stgnn_mta_qsdrop` (decided 2026-10-18,
    user).** The rebuilt data made the trade-off visible: observation dropout costs
    0.25–0.46 s on complete data but saves 6.6–7.4 s when 90% of the network observations
    are missing (§5.1). The paper reports q as the main model (most accurate; the only one
    to beat every baseline incl. GWNet, with the same calibrated intervals as qsdrop,
    §5.7) and qsdrop as the robust deployment variant (§5.1, §5.8). The original L1
    `stgnn_mta` and `stgnn_mta_sdrop` are ablations; the inductive encoder (`v2ind`) is
    the cold-start variant (§5.4).

## 5. Analyses

### 5.1 Missing network data: robustness (test-time only, no retraining; Jiang et al. 2022)
A random share of (bin, segment) observations is hidden, as if fewer buses reported.
Seed 0 of each model, MAE over all test rows (`jobs/analysis.pbs`, 2026-10-17; jobs
27471, 27472, 27478, 27479):

| Hidden | 0% | 25% | 50% | 75% | 90% | Δ at 90% |
|---|---|---|---|---|---|---|
| stgnn_mta (L1, no dropout) | 96.66 | 97.37 | 99.23 | 102.62 | 105.73 | +9.1 |
| stgnn_mta_q (quantiles, no dropout) | 96.58 | 97.30 | 99.12 | 102.03 | 104.94 | +8.4 |
| gwnet_mta (no dropout) | 96.80 | 97.12 | 98.50 | 101.08 | 102.66 | +5.9 |
| **stgnn_mta_qsdrop** (trained with obs. dropout) | 96.96 | **97.06** | **97.25** | **97.61** | **98.32** | **+1.4** |

- The no-graph model scores 98.8 s (3 seeds). Without dropout training, every graph
  model falls *below* no graph once roughly half or more of the observations are
  missing: it was trained only on full-coverage data.
- **Observation dropout during training** (`--train-state-drop 0.9`: each training bin
  hides a random share p ~ U(0, 0.9) of its (step, segment) observations) fixes this.
  - Degradation at 90% hidden falls from +8.4–9.1 s to +1.4 s; the model stays level with
    no graph (98.3 vs 98.8) even with 10% of observations.
  - It is the best model at every coverage below 100%: already at 25% hidden it is
    ahead of q (97.06 vs 97.30) and level with GWNet (97.12).
  - Cost at full coverage: +0.25 s vs the original, +0.46 s vs q (§4, 3 seeds).
  - It leans less on the graph: shuffling the network state costs +25.0 s, vs +34.9 for
    stgnn_mta and +38.4 for q (+20.6 for GWNet).
- GWNet degrades more gracefully than our encoder without dropout (+5.9 vs +9.1 s at
  90%): its adaptive adjacency and node embeddings give it a fallback when observations
  vanish. With dropout, ours is 4.3 s better than GWNet at 90% hidden. GWNet with
  dropout training was not tested.
- Relevance: agencies with partial AVL coverage, feed outages, or newly instrumented fleets.

### 5.2 Permutation importance of the graph model (one input group shuffled on test)
Original `stgnn_mta`, seed 0; MAE over all test rows (base 96.66 s, rebuilt data):

| Group shuffled | MAE (s) | Δ vs 96.66 |
|---|---|---|
| Real-time segment estimate (rt, rt_fresh) | 325.1 | +228 |
| Schedule / distances | 188.7 | +92 |
| **Network state (graph; bins shuffled)** | 131.6 | **+34.9** |
| Time of day | 104.7 | +8.1 |
| Bus state (delays, pace, headway, …) | 103.9 | +7.3 |
| MTA prediction | 103.2 | +6.6 |
| Route id | 97.9 | +1.3 |
| Day of week | 97.8 | +1.2 |
| **Weather + calendar** | 96.8 | **+0.1** |

- Robust variant (qsdrop) for comparison: network state +25.0, rt +241.6, schedule +87.0,
  time +13.5, mta +9.1, bus state +8.3, dow +1.7, route +1.7, weather + calendar +0.3.
- **Weather adds essentially nothing** once the real-time network state is known (both
  models). This goes against common claims in the literature; worth stating.
- Route identity matters little (+1.3 s), so the model generalises across routes rather
  than memorising them.

### 5.3 Evaluation pitfalls: contrast with Rashvand et al. (NYC, arXiv 2303.15495)
- Their setup:
  - 2017 MTA SIRI data, next stop only, a 5-layer MLP;
  - **label = feed `ArrivalTime`, i.e. MTA's expected arrival**;
  - **random 80/20 split**;
  - reported RMSE 35.7 s.
- Our re-implementation of their MLP, next stop only, on our test set:

| Trained on | Error vs MTA's prediction | Error vs actual (GPS) arrival |
|---|---|---|
| MTA predictions (their label) | MAE **33.5 s**, RMSE 80.5 | MAE **43.9 s**, RMSE 72.6 |
| Actual arrivals | MAE 36.6 s | MAE 37.8 s, RMSE 69.1 |

- Training and scoring on MTA's prediction makes the model look **31% better than it
  is**, and it actually learns worse (43.9 vs 37.8 s).
- Our models at h1: 28.3–28.6 s against actual arrivals.
- **Random-split leakage** (`lgbm_mta_randsplit`): the same LightGBM recipe, but train / val
  issues drawn at random from every day Mar 1–Oct 11 (15% of issues, 80.5M rows, about the
  size of the chronological 20% run). The scored test issues are excluded from training, so
  the test rows themselves are never seen, only their neighbours: other buses on the same
  trip and day, and the same days' traffic.
  - Rebuilt data (2026-10-16, job 27401; `results/randsplit*.csv`): scored on the common
    support of the main table, 41 test days, 25.6M rows. The boundary issues of the old
    build are gone (see the bug note below), so no rows need dropping.

| Model | MAE (s) | RMSE | h1 | h2–5 | h6–10 | h11–15 | h16–20 |
|---|---|---|---|---|---|---|---|
| lgbm_mta, chronological split | 99.06 | 163.3 | 28.36 | 57.81 | 97.63 | 130.38 | 157.58 |
| lgbm_mta_randsplit, random split | **96.38** | 157.2 | 28.37 | 57.09 | 95.43 | 126.26 | **151.66** |
| stgnn_mta (ours, chronological, seed 0) | 96.76 | 156.8 | 28.50 | 56.93 | 95.37 | 126.95 | 153.38 |
| stgnn_mta_q (main model, seed 0) | 96.68 | 156.4 | 28.52 | 56.87 | 95.26 | 126.85 | 153.25 |
| stgnn_mta_qsdrop (robust variant, seed 0) | 97.03 | 157.3 | 28.51 | 57.04 | 95.62 | 127.35 | 153.85 |

  - **The random split makes the same model look 2.84 s (2.9%) better**: 99.19 → 96.35 s
    daily mean, better on 100% of 41 days, p = 9e-13. Skill vs MTA rises from +14.7% to
    +17.0%. (Old build: 2.8 s, the same.)
  - **That is enough to erase or reverse the graph advantage.** Evaluated this way, plain
    LightGBM ties the main model (−0.23 s vs stgnn_mta_q, 3 seeds; better on 63% of days,
    p = 0.08) and *beats* the L1 graph model (−0.44 s vs stgnn_mta; 73% of days,
    p = 0.001) and the robust variant (−0.69 s; 83% of days, p = 2e-6). On the old build
    it was a tie with stgnn_mta (−0.18 s, p = 0.1). A paper using a random split would report a tree model
    beating a spatio-temporal GNN, the opposite of the honest ranking (§4: the graph model
    beats LightGBM by about 2 s on 100% of days).
  - The inflation grows with horizon: +0.01 s at h1 (no gain), −5.9 s at h16–20. That is the
    signature of neighbour leakage: issues of the same trip a few minutes apart share most
    of their target stops, and a long-horizon target is one that another training issue
    already observed.
  - Caveat: the random split also trains on the test period itself (the D6 schedule pick and
    school-start traffic). This number is the total effect of both; a day-blocked random
    split (whole days shuffled) would separate them.
  - The validation MAE is not comparable across the splits (random val 90.50 spans all
    months; chronological val 89.9 is August only). Early stopping again never triggered
    (20,000 rounds, best 19,995): the leaked validation set keeps rewarding memorisation.
  - **Bug found and fixed on the way.** The first scoring stopped on the overlap assertion:
    1,404 scored test issues (0.07%) were in `random_train`. Late trips of Aug 31 are stored
    in the Sep 1 sample file; `load("test")` selects by file but the exclusion used
    `service_date`. The fix (commit 0d17a05) matches on the file date, which was checked to
    give 0 overlap. The saved model was rescored with `--predict-only --drop-boundary`,
    which leaves out all 9,399 boundary issues, including the 1,404. (Old build. Files are now
    keyed by service date, so file date = service date and this cannot recur.)
- **Paper point:** the two pitfalls together (wrong label: 31% optimistic; random split: a
  further 2.8 s, more at long horizons, enough to erase or flip the GNN vs tree ranking) are larger than most of the model differences in
  this literature. Our protocol (GPS truth, chronological split, daily paired tests) is a
  contribution in itself.

### 5.4 Cold start: routes never seen in training (BusTr / MAGTTE)
- **Held-out routes:** M102, M7 (high segment sharing), M14D+, M5 (medium), M15+, M42
  (low). They form 401k test issues (about 20% of test).
- Their issues are removed from train and val. At test they get an unknown route id and
  unknown static segment ids for segments never seen in training (48% of their tokens).
  The live network state stays available.
- **Natural novelty is tiny:** the September schedule pick introduced only 8 new stops
  (0.29% of test issues; 0.89% of test targets on unseen segments). Synthetic hold-out
  is therefore needed.
- **Spatial input ablation (BusTr):** with probability 0.3 per training issue, the route
  id and static segment ids are replaced by "unknown".

**Rebuilt data (2026-10-16, Graph WaveNet and transductive seeds 1–2 added 2026-10-18;
`results/coldstart_final*.csv`, 41 days, 5.20M rows on the 6 held-out routes).** All runs
retrained, 3 seeds unless marked; pooled MAE, mean ± sd over seeds:

| Model | All | h1 | h16–20 | Skill vs mta_cal |
|---|---|---|---|---|
| **main model stgnn_mta_q, trained *with* these routes (upper bound)** | 96.64 ± 0.05 | 28.87 | 152.05 | +16.8% |
| stgnn_mta, trained with these routes | 96.82 ± 0.10 | 28.90 | 152.35 | +16.6% |
| gwnet_mta, trained with these routes | 96.84 ± 0.20 | 28.91 | 152.35 | +16.6% |
| qsdrop (robust variant), trained with these routes | 97.08 ± 0.20 | 28.89 | 152.86 | +16.4% |
| seq_mta, trained with these routes | 98.80 ± 0.10 | 29.35 | 155.84 | +14.9% |
| lgbm_mta, trained with these routes (1 run) | 99.02 | 28.77 | 156.38 | +14.7% |
| **inductive graph, held out + ablation** (`v2ind_ho_abl`) | **101.30 ± 0.36** | **29.87** | **158.80** | **+12.8%** |
| inductive, segment length only (1 seed) | 101.30 | 29.88 | 158.73 | +12.8% |
| inductive, no segment features (1 seed) | 101.54 | 29.86 | 159.33 | +12.6% |
| inductive final recipe (quantiles + obs. dropout, `v2indqsd`) | 101.90 ± 0.19 | 29.95 | 159.87 | +12.3% |
| seq, held out + ablation | 103.51 ± 0.20 | 30.43 | 162.81 | +10.9% |
| LightGBM, held out (1 run) | 105.20 | 30.51 | 164.31 | +9.4% |
| Graph WaveNet, held out + ablation (`gwnet_mta_ho_abl`) | 109.30 ± 1.13 | 31.25 | 171.80 | +5.9% |
| transductive graph (ours), held out + ablation | 111.21 ± 1.24 | 31.16 | 176.01 | +4.2% |
| mta_cal | 116.14 | 40.79 | 180.32 | 0 |

Paired daily MAE, seeds averaged (41 days; Wilcoxon):

| Comparison | ΔMAE (s) | Days better | p |
|---|---|---|---|
| inductive vs seq + ablation | −2.29 | 98% | 3e-11 |
| inductive vs LightGBM held out | −4.03 | 100% | 9e-13 |
| inductive vs transductive graph held out (3 seeds) | −9.83 | 100% | 9e-13 |
| inductive vs Graph WaveNet held out | −7.86 | 100% | 9e-13 |
| Graph WaveNet held out vs transductive graph held out | −1.97 | 95% | 2e-11 |
| Graph WaveNet held out vs seq + ablation | +5.57 | 2% | 3e-12 |
| Graph WaveNet held out vs trained with them | +12.37 | 0% | 9e-13 |
| inductive final recipe (qsd) vs inductive L1 | +0.61 | 10% | 1e-8 |
| qsd vs seq + ablation | −1.67 | 93% | 1e-10 |
| qsd vs LightGBM held out | −3.42 | 100% | 9e-13 |
| length only vs all attributes | −0.01 | 46% | 0.8 |
| no features vs all attributes | +0.25 | 24% | 0.003 |
| no features vs seq + ablation | −2.04 | 100% | 9e-13 |
| inductive (held out) vs stgnn_mta trained with them | +4.52 | 0% | 9e-13 |
| inductive (held out) vs main model q trained with them | +4.69 | 0% | 9e-13 |
| LightGBM held out vs trained with them | +6.21 | 0% | 9e-13 |

- **Graph WaveNet fails on unseen routes, as predicted** (2026-10-18, job 27480): 109.30 s,
  5.6 s *worse* than no graph (better on 2% of days) and 12.4 s worse than when trained
  with the routes. Its learned node embeddings and adaptive adjacency are transductive,
  like ours; it fails slightly less badly than our transductive encoder (−2.0 s, 95% of
  days). Both transductive encoders vary a lot across seeds on unseen routes (sd 1.1–1.2 s
  vs 0.2–0.4 s for the others): what untrained embeddings do is luck. The inductive encoder
  beats Graph WaveNet by 7.9 s on 100% of days. So the cold-start result is not specific to
  our encoder: **any encoder with per-node learned identity fails to transfer; removing it
  fixes transfer.**
- The transductive graph now has 3 seeds: 111.21 ± 1.24 s (seed 0 alone: 110.09).
- Every conclusion below survives the rebuild: inductive graph < seq + ablation < LightGBM
  held out < Graph WaveNet held out < transductive graph < mta_cal; message passing over anonymous segments is what
  transfers (no features still beats no graph by 2.0 s on 100% of days); signals and bus
  lanes add nothing beyond segment length (now an exact tie, p = 0.8).
- **Changed:** the final recipe is no longer free on cold start. It costs 0.61 s vs the L1
  inductive model (worse on 90% of days), whereas the old build had a tie (+0.11 s, p = 0.5).
  It still beats every non-inductive alternative. So for a changing network the
  recommendation is the L1 inductive model when point accuracy matters, the quantile one
  when intervals are needed.
- The tables and numbers below are from the **old build** (42 days) and include runs that
  were not retrained (no-ablation and v2-transductive variants); they are kept for the
  history of the argument.

Results on the 6 held-out routes, old build (MAE, s):

| Model | All | h1 | h16–20 | Skill vs mta_cal |
|---|---|---|---|---|
| stgnn_mta, trained *with* these routes (upper bound) | 96.5 | 28.9 | 151.9 | +16.5% |
| seq_mta, trained with these routes | 98.2 | 29.2 | 155.3 | +15.1% |
| **inductive graph, held out + ablation** (`stgnn_mta_v2ind_ho_abl`) | **101.1** | **29.8** | **158.7** | **+12.5%** |
| seq, held out + ablation | 103.0 | 30.5 | 162.4 | +10.9% |
| seq v2 (school + attrs), held out + ablation | 103.3 | 30.5 | 162.3 | +10.6% |
| stgnn v2 (school + attrs as token inputs), held out + ablation | 111.0 | 31.1 | 176.4 | +4.0% |
| stgnn, held out + ablation | 110.6 | 31.2 | 176.2 | +4.3% |
| stgnn, held out, no ablation | 115.3 | 36.1 | 183.6 | +0.3% |
| seq, held out, no ablation | 116.8 | 36.5 | 185.0 | −1.0% |
| mta_cal | 115.6 | 40.7 | 179.9 | 0 |

- **BusTr's ablation is essential:** it gains 7.6–13.8 s on unseen routes.
- **Negative result: our graph encoder hurts on unseen routes** (110.6 vs 103.0 s). Its
  per-segment learned vectors and learned adaptive adjacency memorise segment identity,
  and they are untrained for unseen segments. The ablation removed route and static ids
  but not these encoder identities.
- **Fix: an *inductive* encoder** (`--inductive`). Segments are described by
  physical attributes (signals, bus lanes, length) instead of learned vectors, with no
  adaptive adjacency, in the spirit of MAGTTE's inductive learning.
  - **It turns the negative result around: 101.1 s on unseen routes**, vs 110.9 for the
    transductive graph (−9.8 s) and 103.0 for no graph (−1.9 s). Paired daily: better
    on **100% of the 42 days** in both comparisons (p = 5e-13). The gain over no graph
    grows with horizon (−0.7 s at h1, −3.7 s at h16–20), the same signature as the graph
    gain on seen routes.
  - The unseen-route penalty shrinks from 14.4 s (110.9 vs 96.5) to 4.6 s.
  - Price on seen routes: 97.2 vs 96.6 s (+0.6 s, 95% of days worse). That is on par
    with GWNet (97.1), better than DCRNN (97.5) and 1.3 s better than no graph. So:
    transductive encoder for a fixed network, inductive for a changing one; or a hybrid
    (future work).
  - Giving the *transductive* model the same attributes as token inputs does not help
    unseen routes (111.0 vs 110.6, n.s.). The fix is removing per-segment identity from
    the encoder, not merely adding attributes.
  - **3 seeds:** inductive 101.22 ± 0.10 s vs seq (held out + ablation) 103.04 ± 0.04
    s, so the 1.8 s gap is about 20× the seed spread.
  - **Controls: the physical attributes are not what transfers.**

| Inductive encoder sees | Unseen-route MAE (s) | vs all attributes (paired) |
|---|---|---|
| all (length, signals, signals/km, bus-lane shares) | 101.22 (3 seeds) | – |
| segment length only (`--inductive-feats len`) | 100.95 | −0.21 s, p = 0.006 |
| nothing, a constant (`none`) | 101.40 | +0.25 s, p = 2e-4 |
| (no graph: seq, held out + ablation) | 103.04 | |

    Even with *no* segment information, the inductive graph beats no graph by 1.6 s
    (better on 90% of days, p = 2e-10). **The cold-start gain comes from message passing
    over anonymous segments,** i.e. from live upstream / downstream speeds on shared
    streets, not from signals or bus lanes. The learned per-segment identity is what
    breaks transfer. Signals and bus lanes add nothing beyond segment length.
- **LightGBM held out** (same 6 routes removed from training; unseen `stop_id` /
  `route_id` categories fall to LightGBM's default branch; no ablation trick):
  - 105.0 s on the held-out routes, vs 98.9 when trained with them (+6.0 s; worse on
    100% of days);
  - path attributes (`--attrs`) recover 0.19 s (104.8; 86% of days, p = 1e-7). This is
    the only model where the physical attributes measurably help transfer, and the
    effect is still small;
  - the inductive graph beats held-out LightGBM by 4.0 s on 100% of days (p = 5e-13),
    with the gap growing from 0.7 s at h1 to 5.5 s at h16–20.
  - Cold-start ranking: inductive graph 101.1 < seq + ablation 103.0 < LightGBM 104.8–105.0
    < transductive graph 110.6 < mta_cal 115.6.
- **Cold start with the final recipe** (`stgnn_mta_v2indqsd_ho_abl`: inductive + quantiles
  + observation dropout, 3 seeds; `results/coldstart_final*.csv`):
  - 101.32 ± 0.05 s on the held-out routes, a tie with the L1 inductive model (101.22;
    +0.11 s, p = 0.5). The final recipe costs nothing on cold start. (Rebuilt data: it
    does, +0.61 s; see the rebuilt table above.)
  - Seed-averaged paired: beats no graph + ablation by 1.8 s (95% of days, p = 6e-12),
    held-out LightGBM by 3.75 s and the transductive graph by 9.2 s (both 100% of days).
  - Its 90% intervals cover 88.8–89.0% (all routes, width about 385 s), so the ranges
    carry over to the inductive variant.

### 5.5 School calendar, traffic signals and bus lanes
- Segment attributes check against reality:
  - M15/M15+ on 1st/2nd Ave are about 70–80% bus lane at ~10–11 signals/km;
  - M7 (Columbus/Amsterdam) has about 9% bus lane;
  - M14D+ has 58% of its lanes in force at 8 AM vs 12% at 3 AM.
- Expected effects:
  - **Static attributes: about zero** on seen segments, since learned segment embeddings
    already absorb them;
  - useful for unseen segments and for interpretation (seconds per signal);
  - **Part-time lane activity** varies by hour and cannot be absorbed by a static
    embedding, so it is the most likely to help.
- **Results on the main test (old build, not rerun; seed 0, paired daily, 42 days):**

| Comparison | ΔMAE (s) | Days better | Wilcoxon p |
|---|---|---|---|
| stgnn v2 (school + attrs) vs v1 | −0.01 | 45% | 1.0 |
| stgnn v2 vs v2 without attrs (attrs only) | −0.04 | 69% | 0.03 |
| stgnn v2 without attrs vs v1 (school flag only) | +0.03 | 36% | 0.2 |
| seq v2 vs seq v1 | +0.28 | 26% | 3e-4 |

  - **Neither the school flag nor signals / bus lanes (incl. hour-of-week lane activity)
    improves accuracy.** All differences are below the 0.1–0.2 s seed spread. The tiny
    attribute effect (−0.04 s) is not practically meaningful. The seq v2 loss (+0.28 s)
    is about one seed s.d., so it should not be over-read either.
  - Interpretation: the live network state already measures what the static attributes
    describe. Their value is for transfer (§5.4), not for accuracy on known segments.
- **School-start natural experiment** (no school Sep 1–9, school from Sep 10):
  - Every model is about 14% worse from Sep 10: `mta_cal` 104.2 → 118.7 s, ours 86.7 →
    99.2, seq 87.8 → 101.2. Mean travel time to target rises 759 → 784 s.
  - Skill vs MTA is unchanged (16.8% vs 16.4%), so the regime change hits all predictors
    equally; it is harder traffic, not model failure.
  - The school flag doesn't help even after Sep 10 (−0.03 s): training (Mar–Jun) already
    contains school days, and the network state reflects the extra traffic directly.
- **LightGBM** (paired daily vs `lgbm_mta`):
  - `+school`: −0.14 s (74% of days, p = 2e-4), all of it on school days (−0.20 s from
    Sep 10; +0.03 s before);
  - `+attrs` (path signals, bus-lane shares, lanes active now): −0.03 s (p = 0.01);
  - `+both` (`lgbm_mta_school_attrs`): −0.14 s vs base (76% of days, p = 7e-4), identical
    to `+school` alone (+0.01 s, p = 0.7). The attributes add nothing on top of school.
  - Statistically detectable, practically negligible; the same conclusion as for the
    neural models.

### 5.6 Error analysis by operating condition (`python -m bus_eta.error_analysis`)
MAE (s) on the common support of the compared models (seed 0), all horizons, rebuilt data
(41 days, 25.6M rows). Skill is vs `mta_cal`; graph gain is seq_mta − stgnn_mta. Files:
`results/error_analysis.csv`, `error_analysis_by_route.csv`.

| Condition | Rows | mta_cal | MTA published | lgbm | seq | ours (stgnn_mta) | Ours skill | Graph gain |
|---|---|---|---|---|---|---|---|---|
| Night 0–6 | 2.2M | 83.0 | 82.8 | 66.0 | 66.3 | 65.5 | 21.2% | 0.9 |
| AM peak 6–10 | 5.4M | 107.1 | 113.6 | 89.6 | 89.7 | 87.9 | 17.9% | 1.7 |
| Midday 10–16 | 8.2M | 131.4 | 134.3 | 114.4 | 113.9 | 111.1 | 15.5% | 2.8 |
| PM peak 16–20 | 5.7M | 123.5 | 123.0 | 106.6 | 106.5 | 104.2 | 15.6% | 2.3 |
| Evening 20–24 | 4.1M | 104.5 | 102.1 | 87.9 | 87.7 | 86.0 | 17.7% | 1.7 |
| Weekday | 19.3M | 118.6 | 120.2 | 100.8 | 100.8 | 98.9 | 16.6% | 1.9 |
| Weekend | 6.3M | 108.4 | 110.6 | 93.7 | 92.9 | 90.3 | 16.8% | 2.7 |
| Dry | 20.9M | 117.6 | 119.5 | 100.3 | 100.3 | 98.3 | 16.4% | 2.0 |
| Light rain (<1 mm/h) | 3.1M | 112.0 | 113.5 | 96.1 | 95.2 | 92.1 | 17.8% | 3.1 |
| Heavy rain (≥1 mm/h) | 1.6M | 104.5 | 105.5 | 88.5 | 87.9 | 85.9 | 17.8% | 2.1 |
| SBS routes | 2.6M | 106.5 | 106.2 | 88.5 | 88.9 | 86.9 | 18.4% | 2.0 |
| Local routes | 23.0M | 117.2 | 119.2 | 100.3 | 100.0 | 97.9 | 16.5% | 2.1 |
| Service alert live on route | 16.8M | 120.5 | 122.7 | 103.3 | 102.9 | 100.6 | 16.5% | 2.3 |
| No alert | 8.8M | 107.7 | 108.6 | 91.0 | 91.1 | 89.4 | 17.0% | 1.8 |
| Before school (Sep 1–9) | 5.2M | 104.7 | 105.8 | 88.6 | 88.2 | 86.9 | 17.1% | 1.3 |
| School (Sep 10+) | 20.4M | 119.0 | 120.9 | 101.7 | 101.6 | 99.3 | 16.6% | 2.3 |

- **The gain is uniform.** Skill vs MTA is 15.5–21.2% in every time period, day type,
  weather and route type. **All 41 routes improve** (skill 9.6–33.6%). The graph gain
  is positive on every route (1.1 s on M14D+, M35, M10 up to 4.3 s on M20, 3.9 on M12,
  3.7 on M66). Old build: ≥ 0, with M35 at 0.0.
- **Midday, not the peaks, is hardest** for every predictor (ours 111 s vs 88 / 104 s at
  the AM / PM peaks). It is also where the graph helps most (2.8 s): long dwell times,
  deliveries and curb conflicts make downstream conditions matter more.
- **The evening is no longer an outlier.** Old build: LightGBM was 3 s worse than seq at
  20–24 h (91.1 vs 88.1), an artefact of the UTC split; now they are level (87.9 vs 87.7),
  and the evening graph gain rose from 1.3 to 1.7 s.
- **MTA's published ETA is worst in the AM peak** (−6.1% vs the calibrated version):
  its bias is time-dependent. Our models need no such correction.
- **Rain does not make prediction harder.** Rainy hours have *lower* error for all
  models; this is confounded with time of day, so it is not causal. It agrees with the
  near-zero weather permutation importance (§5.2). The graph gain is largest in light
  rain (3.1 s).
- **SBS routes** (off-board fare, bus lanes) are easier (86.9 vs 97.9 s) and have higher
  skill.
- **Service alerts** are live on 66% of route-issues. The MTA texts are generic ("You may
  wait longer…"). Error is about 11 s higher when an alert is live, for every model.
  Skill drops only slightly (16.5 vs 17.0%). Alerts mark harder conditions but carry
  little specific information; a free-text alert feature is low priority.
- **School start:** error rises for every model after Sep 10 (99.3 vs 86.9 s for ours),
  and so does the graph gain (2.3 vs 1.3 s). Skill is stable (16.6 vs 17.1%).

### 5.7 Uncertainty: quantile head (`--quantiles 0.05 0.1 0.25 0.5 0.75 0.9 0.95`)
Same graph model, one output per quantile, pinball loss; the median is the point ETA.
Scored with `python -m bus_eta.uncertainty` (`results/uncertainty_final.csv`; rebuilt
data, all 26.3M test rows, 3 seeds each).

- **The point accuracy improves.** stgnn_mta_q is the most accurate model on the rebuilt
  data: 96.61 s, −0.22 s vs the L1 original (85% of days, p = 1e-8), −0.25 s vs GWNet (§4).
  The same head on GWNet doesn't improve it (+0.09 s, §4 finding 4).
- **Calibration** (all horizons, mean of 3 seeds): share of arrivals below q5 / q10 /
  q25 / q50 / q75 / q90 / q95 is 4.9 / 9.7 / 24.2 / 48.5 / 72.9 / 88.4 / 93.9% for
  stgnn_mta_q, and 4.8 / 9.5 / 23.8 / 47.7 / 72.4 / 88.0 / 93.6% for qsdrop. Quantiles
  cross on ≤ 0.01% of rows.

| Horizon | 50% int. coverage | 80% int. coverage | 90% int. coverage | 90% width (s) |
|---|---|---|---|---|
| h1 | 50.2 | 80.4 | 90.3 | 117 |
| h2–5 | 49.2 | 79.3 | 89.5 | 229 |
| h6–10 | 48.6 | 78.6 | 88.9 | 375 |
| h11–15 | 48.2 | 78.1 | 88.5 | 492 |
| h16–20 | 48.0 | 77.8 | 88.3 | 588 |
| all | 48.7 | 78.7 | 89.0 | 378 |

  (Main model stgnn_mta_q, mean of 3 seeds; qsdrop is within 0.4 pt everywhere, e.g.
  88.8% / 378 s for the 90% interval overall. Coverage of the 50% / 80% intervals is the
  difference of the shares below q75 − q25 and q90 − q10.)
- Slight under-coverage that grows with horizon (−2.3 pt for the 80% interval at
  h16–20; slightly more than the old build's −1.2) and a median biased a little early
  (47.7% below q50). A typical test-period shift (school start).
- **Conformal recalibration** (2026-10-18; `python -m bus_eta.conformal`,
  `results/conformal*.csv`). Split CQR (Romano et al. 2019), calibrated separately for each
  horizon on the validation month (Aug; predictions from `stgnn --eval-only <m> --split
  val`), scored on the same test rows; mean of 3 seeds:

| Interval | Band | Raw coverage | CQR coverage | Raw width (s) | CQR width (s) | Interval score raw → CQR |
|---|---|---|---|---|---|---|
| 90% | h1 | 90.3 | 89.7 | 117 | 115 | 170.5 → 170.5 |
| 90% | h6–10 | 88.9 | 89.5 | 375 | 381 | 539.8 → 539.3 |
| 90% | h16–20 | 88.3 | 89.1 | 588 | 602 | 863.9 → 862.0 |
| 90% | all | 89.0 | 89.5 | 378 | 385 | 548.3 → 547.6 |
| 80% | all | 78.7 | 79.3 | 288 | 293 | 447.7 → 447.4 |
| 50% | all | 48.7 | 49.4 | 148 | 151 | 311.7 → 311.7 |

  - Recalibration closes about half the coverage gap for 1.8% wider intervals (+7 s at
    90%), and the gap no longer grows as much with horizon (h16–20: 88.3 → 89.1%). It
    narrows h1, which was over-covered. The interval score improves only slightly.
  - **The raw intervals are already close to calibrated.** On validation they cover
    89.5% (90% interval; 89.2% at h16–20), so the corrections are small (mean offset
    4 s at 90%, from −0.8 s at h1 to +6.8 s at h16–20).
  - The remaining 0.5-pt shortfall is drift from the validation month to the test month
    (test covers 0.5 pt less than validation before recalibration). Split conformal
    assumes the two are exchangeable, which a time split breaks. Online / adaptive
    conformal (e.g. ACI, Gibbs & Candès 2021) would track it; future work.
  - qsdrop: the same picture (90%: 88.9 → 89.4%, width 378 → 384 s).
  - Paper: report the raw intervals as the main result (near-nominal without any
    post-processing) and CQR as a cheap fix that brings every band within 0.9 pt of nominal
    for the 90% interval (1.2 pt for the 80% interval at h16–20).
- **Adaptive intervals (old build, not rerun):** split into fifths by 80% interval width,
  the mean absolute error was 33 / 61 / 87 / 116 / 184 s with coverage 80.3–78.7% in every
  fifth; Spearman(width, |error|) = 0.51. The model knows which ETAs are unreliable.
- Use case: "arriving in 6–9 min" displays, and transfer / missed-connection risk.
  MTA publishes only a point ETA.

### 5.8 Robust variant: quantile head + observation dropout (`stgnn_mta_qsdrop`)

| Hidden network observations | 0% | 25% | 50% | 75% | 90% |
|---|---|---|---|---|---|
| qsdrop: median MAE (s, seed 0) | 96.96 | 97.06 | 97.25 | 97.61 | 98.32 |
| qsdrop: 90% interval coverage (%) | 88.7 | 88.7 | 88.8 | 88.8 | 88.9 |
| qsdrop: 90% interval mean width (s) | 374 | 375 | 377 | 379 | 383 |
| main model q: median MAE (s, seed 0) | 96.58 | 97.30 | 99.12 | 102.03 | 104.94 |
| main model q: 90% interval coverage (%) | 88.7 | 89.1 | 89.2 | 88.5 | 87.0 |
| main model q: 90% interval mean width (s) | 373 | 384 | 394 | 395 | 384 |

(`results/uncertainty_drop.csv`, `results/uncertainty_drop_q.csv`.)

- **qsdrop's intervals stay calibrated when data go missing.** Widths grow slightly (+2%)
  instead of coverage collapsing.
- **The main model's intervals hold up too, but less well** (2026-10-18): coverage stays
  within 2 pt of nominal, but drops to 87.0% at 90% hidden while the width *shrinks*
  (395 → 384 s) even though the error grows by 8.4 s. Without dropout training the model
  doesn't know it is missing data. Interval quality is therefore not a reason to prefer
  either model on complete data; robustness is.
- 3 seeds: MAE 97.06 ± 0.20 s, +16.4% vs `mta_cal`; 90% coverage 88.7 / 88.7 / 89.2%
  (mean width 374 / 374 / 384 s).
- **Rebuilt data: the combination is no longer free.** Old build: it tied the original
  (+0.06 s, p = 0.1). Now: +0.25 s vs the original and +0.46 s vs quantiles alone; the
  cost comes from observation dropout (sdrop alone: +0.24 s) and is concentrated at long
  horizons (+0.76 s vs GWNet at h16–20). In exchange it is the best model whenever ≥ 25%
  of observations are missing (§5.1). Hence its role as the robust variant, not the main
  model (§4 finding 12).
- Cold start (§5.4): its inductive variant costs 0.61 s vs the L1 inductive model.

### 5.9 Paper figures (`python -m bus_eta.figures`, → `docs/figures/*.pdf|png`)
Numbers are cached in `results/figdata/<figure>.csv`; restyling needs no recomputation.
Neural models are averaged over 3 seeds unless stated. All six regenerated on the
rebuilt data (`--recompute`, 2026-10-17) and checked visually. "Ours" is the main model
`stgnn_mta_q` (`MAIN` in `figures.py`; switched from qsdrop and regenerated 2026-10-18).

- `horizon`: MAE and skill vs `mta_cal` for h = 1..20, all main baselines. Skill peaks at
  h1 (about 26%, where LightGBM is best) and settles at 15–16% from h6 on. The graph models
  fan out above no graph as h grows. The historical average is +12.7% at h1 and ≤ 0 from
  about h7 on.
- `robustness`: §5.1 as a curve (seed 0 of the main model q, Graph WaveNet and the robust
  variant qsdrop, labelled "Ours + obs. dropout in training"), with the no-graph model (98.8 s, 3 seeds) as a horizontal line. Values are on the common
  support, so they differ from the §5.1 table by < 0.2 s.
- `permutation`: §5.2 as bars, log scale (original `stgnn_mta`, seed 0: the same encoder
  as the main model with L1 loss; only the graph group was shuffled for q, +38.4 s).
- `tod`: MAE by issue hour for the main model, no graph, LightGBM and `mta_cal`, plus the
  graph gain. The gain is largest from 11:00 to 17:00 (2.2–3.6 s) and smallest at
  02–05 (−0.2–0.7 s), when few other buses are on the road to observe. **The old dip at
  19:00 and the LightGBM spike at 20:00 are gone** (they were the UTC-split artefact);
  the curves are smooth through the evening.
- `map`: graph gain per segment (no-graph minus ours, seed-averaged), over every prediction
  whose target stop is reached through that segment, all horizons, ≥ 500 predictions.
  1,945 of 2,111 segments qualify: the graph helps on 98.1% (median +1.8 s), and only 0.5%
  are worse by more than 1 s. The plot is turned 29° so Manhattan's grid runs upright.
- `trip`: one departure, chosen automatically as the weekday 16–20 h departure with all 20
  horizons whose mean error for ours is nearest the median (not hand-picked): now M104,
  Sep 7, 16:15 (with qsdrop: M102, Sep 16, 18:00; old build: M11). Actual vs ours (with
  the 5–95% interval), no graph and MTA. All three predict a little early beyond h12;
  the interval covers the actual arrival at 19 of 20 stops. Caveat: Sep 7 was Labor Day,
  which the `dow <= 5` filter counts as a weekday; excluding holidays would pick another
  trip.

## 6. Related work notes (papers in `research papers/`, kept local, not in git)

| Paper | Key point | How we use it |
|---|---|---|
| Petersen, Rodrigues, Pereira 2019, ESWA (ConvLSTM) | Multi-output link travel time, Copenhagen; beats the deployed Movia system | Closest in spirit (beat the operator). Baseline to re-implement (TODO) |
| Yuan et al. 2020, Electronics (wide & deep + attention RNN) | Bus travel time, Guangzhou/Shenzhen | Related work; possible baseline |
| Ma, Chan, Rajasegarar, Leckie 2022, ESWA (MAGTTE) | Multi-attention GNN; routes with sparse or no data; inductive learning | Motivates the cold-start test and the inductive encoder |
| Rashvand et al. 2023/24, arXiv 2303.15495 | NYC next-stop MLP, RMSE 35.7 s | Same city. Label and random-split pitfalls (§5.3) |
| Barnes et al. 2020, KDD (BusTr, Google) | Bus delays from road traffic; spatial input ablation; generalisation to new routes and areas; repeated runs with t-tests | Ablation trick (§5.4), evaluation style; traffic was their key input (not open for NYC streets) |
| Jiang et al. 2022, Promet (hybrid incomplete data) | LSTM with missing GPS / smart-card data | Missing-data robustness (§5.1) |
| Hafizi et al. 2025, Transp. Res. Procedia | Tehran; neural-network sensitivity analysis | Permutation importance (§5.2) |
| Cheon & Hwang 2026, JKSCI | Transformer summing per-section times | Related work (additive / monotone ETAs; not needed, §4.9) |
| Liu & Liu 2024, AI (GMTP) | GATv2 + BERT-style self-supervised pretraining | Future work: encoder pretraining |
| Ding et al. 2025, arXiv (GSMT) | GAT + seq2seq + speed-regime corrector, Kuala Lumpur | Related work |
| Drosouli et al. 2023, Sensors (ST-GCRN) | GCN+LSTM for metro / bike flow | Related work (graph forecasting) |
| Li et al. 2018 (DCRNN); Wu et al. 2019 (Graph WaveNet) | Standard spatio-temporal graph baselines | Implemented as encoders (§4) |

## 7. Limitations (to state in the paper)
- One city, and one 7-month period. The test period is autumn only.
- The ground truth depends on GPS ping density: 4% of labels have a gap > 60 s.
- The main model's lead over Graph WaveNet is small (0.25 s; 0.34 s when GWNet gets the
  same quantile head, which doesn't help it). The L1 versions tie. The contribution is
  the setup and graph, not a radically new network.
- Without observation dropout the graph models degrade below about 50% coverage
  (§5.1); with it, accuracy on complete data drops by 0.25–0.46 s.
- Our encoder with L1 does not beat Graph WaveNet (§4 finding 4); only with the quantile
  loss does it pull ahead, by 0.25–0.34 s. The contribution is the protocol, the graph gain,
  robustness and cold start, not the encoder.
- Bus-lane direction isn't matched (parallel heading only); OSM signal completeness is
  not verified.
- Scored on a fixed 25% hash sample of test issues (25.6M predictions over all 41 days);
  see the EVAL_FRAC note in §8.
- **UTC day boundary (found 2026-10-16 from the `tod` figure; fixed the same day and
  everything rebuilt and retrained on branch `fix-utc-day-boundary`, see §2 "Data rebuild").**
  The description below is of the old build (`data/archive_utc/`).
  - The GTFS-rt archive is stored in daily files by UTC date. `arrivals` projects each
    file on its own, so a trip running across 00:00 UTC (20:00 EDT, 19:00 EST) is split
    in two, each half without the other's arrivals. That is 3.6–4.1% of trips per day
    (checked Sep 15/16 and Sep 19/20).
  - `samples.build_day` takes targets only from the same day's file. Targets after 00:00
    UTC are lost for issues raised before it: issues at 19:45–20:00 EDT average h = 4.1
    instead of 8.8, and hour 19 has 10.6 rows per issue instead of about 12.8. That is
    roughly 0.2M of 26M test rows (about 1%).
  - Issues just after 00:00 UTC lose their own trip's earlier history (bus-state
    features), which likely explains LightGBM's +8 s spike at 20:00 (ours +0.4 s
    relative to its neighbours).
  - All models are scored on the same rows, so comparisons are unaffected; absolute MAE
    near 19–20 h is slightly biased (fewer long horizons).
  - Fix (user decision 2026-10-16: fix and retrain everything): arrivals are keyed by
    service date, built from the archive files of d and d+1; see §2 "Data rebuild".

### Code review (2026-10-17, after the rebuild)
Every stage was read end to end: `static_gtfs`, `geo`, `arrivals`, `graph`, `network_state`,
`samples`, `features`, `context`, `stgnn_data`, `stgnn` (all encoders, training, eval-only),
`lgbm`, `baselines`, `hist_avg`, `rashvand_mlp`, `evaluate`, `paired`, `uncertainty`,
`error_analysis`. Checked and fine:
- **Causality.** Targets need `arr_seen > t_issue`. Other buses' runs and dwells, headways and
  MTA predictions are as-of joins on the time they were revealed. The network state files each
  traversal under the bin of its revealing ping, and the model reads only bins that ended
  before the issue bin. GPS projection is a forward filter (uses only past pings). Open-Meteo's
  hourly precipitation is the sum of the *preceding* hour, so joining on the issue hour is
  causal.
- **Splits.** Normalisation statistics and `mta_cal` fitted on train only; early stopping on
  validation only; the scored test issues are a fixed hash sample shared by every model (the
  LightGBM runs train on exactly the neural models' 76.35M rows); held-out routes are removed
  from train and val, and route index 0 is the "unknown" id.
- **Keys.** trip_id is unique across the four schedule picks (136,310 ids), and
  (trip_id, stop_idx) is unique in `stop_times`, so the trip_id-only joins are safe.

Issues found (none changes a conclusion; no rerun):
1. **MTA availability uses the future.** `samples.mta_predictions` keeps an MTA prediction if
   the stop's *last* appearance in the day's feed is ≥ t_issue − 60 s, i.e. a whole-day maximum.
   The value itself is causal (the latest snapshot at or before t_issue), but the
   "MTA still predicting this stop" mask can look ahead. Measured on Sep 16: 1.07% of the
   accepted MTA inputs come from a snapshot older than 60 s (0.36% > 120 s, 0.18% > 300 s),
   and 0.3% of rows are rejected. Impact: MTA input is worth 0.11 s with the graph (§4), and
   the same rows are scored for every model, so the effect on any comparison is far below
   seed noise. Fix (next rebuild): accept only if the matched snapshot is ≤ 60 s old.
2. **Cold start is "unseen in labels, not in sensing".** The held-out routes' issues are
   removed from training, but their buses still feed the network state in every month, so
   segments used only by those routes have observed live speeds during training. This is a
   protocol choice (a tracked route that has no ETA model yet), and the paper must state it.
   A stricter variant (also hide their traversals from the training state) is future work.
3. Minor: the error-analysis rain groups use the previous hour's precipitation (fine as a
   feature, slightly lagged as a condition); one of 2,111 segments is shorter than the 25 m
   departure/arrival threshold gap.

## 8. Open items / TODO
- [x] DCRNN seeds 0–1, GWNet seed 2 (in table).
- [x] Random-split pitfall result (LightGBM): +2.8 s optimistic, ties our graph model (§5.3).
  Rebuilt data: +2.84 s, and it now *beats* our graph model by 0.4–0.7 s (p ≤ 0.001).
- [ ] Optional: day-blocked random split, to separate neighbour leakage from training on the test period.
- [x] Cold-start: inductive encoder; segment attributes.
- [x] Cold-start: inductive controls (len / none); seeds 1–2.
- [x] Cold-start: LightGBM held out (with and without attributes).
- [x] nn_v2: school + attributes on the main test (no gain).
- [x] Headline model decided (2026-10-15, user): the combined recipe `stgnn_mta_qsdrop`
  (quantiles + observation dropout) is the paper's main model. **Superseded 2026-10-18**
  (see below). The original `stgnn_mta`
  stays as an ablation, and the inductive encoder is the cold-start variant.
- [x] Service alerts: not used as a model input (decided 2026-10-15). They are live on 66%
  of route-issues, their text is generic and the cause / effect fields are empty, so they
  only mark harder periods (§5.6). The paper reports them in the error analysis and as a
  limitation of open alert data.
- [x] Training with random state dropout (robustness fix, §5.1).
- [x] Error analysis: rush vs off-peak, rain, route, SBS vs local, school start, service alerts (§5.6).
- [x] Uncertainty (quantile) outputs and calibration (§5.7).
- [x] Conformal recalibration (2026-10-18, §5.7): split CQR per horizon, 90% coverage
  89.0 → 89.5% for +7 s width; the rest is val → test drift. Optional: adaptive conformal.
- [x] Petersen ConvLSTM baseline (finding 10), incl. the 20-epoch check (use e20 in the paper).
- [x] Final protocol (started and scored 2026-10-15; batch size 2048, as in seed 0); all
  runs done, `dcrnn_mta` s2 (27335) last, at 11:35:
  - main model `stgnn_mta_qsdrop` seeds 1–2 (27324, 27327);
  - ablation seeds `stgnn_mta_q` / `stgnn_mta_sdrop` s1–2;
  - cold start with the final recipe, `stgnn_mta_v2indqsd_ho_abl` (inductive + quantiles
    + dropout, nn_v2, held out + ablation 0.3), seeds 0–2 (27330–27332);
  - `convlstm_mta_e20` s1–2 and `dcrnn_mta` s2 (27333–27335).
  - Scoring stays on EVAL_FRAC = 0.25 (25.6M rows on the rebuilt data): significance is
    computed over days, so a 4× larger sample of the same 41 days adds almost no power. It also keeps every
    side analysis valid, e.g. the random-split model, which trained on the other 75%.
- [x] Historical-average baseline (finding 11).
- [x] T-GCN / A3T-GCN baselines (finding 4): `a3tgcn_mta` seeds 0–2 (27347–27349),
  `tgcn_mta` seed 0 (27350).
- [x] STAEformer baseline (Liu et al., CIKM 2023), encoder `--encoder staeformer`, 3 seeds.
  Seeds 0–1 (27352/27353, GPU 1) crashed in epoch 8 of 10 on 2026-10-15 at 19:13 with
  "uncorrectable ECC error": GPU 1 shows 4 uncorrected DRAM errors and a row remap pending,
  which needs an admin GPU reset. Best checkpoints (epoch 7, val 88.79 / 89.47 s) kept as
  `staeformer_mta_ep7.pt` / `staeformer_mta_s1_ep7.pt`, but they are mid-schedule
  (OneCycle LR not annealed), so not used for the paper. `jobs/stgnn.pbs` now skips GPUs with uncorrected ECC errors. A retry pinned to GPU 1
  (27383 / 27384, 21:35) failed on the first host-to-GPU copy with the same ECC error, so
  GPU 1 rejects all CUDA work until reset. Seeds 0 and 1 rerun back to back on GPU 0 as
  27385 (`SEEDS="0 1"`), done 2026-10-16 04:15. Scored in `final` (finding 4). Seed 2 (27371, GPU 0) is
  unaffected, at about 22 min/epoch.
- [x] Figures (§5.9): `horizon`, `robustness`, `permutation`, `tod`, `map`, `trip`.
- [x] UTC day-boundary split of trips (§7) and the DST start_date bug: fixed in `arrivals` /
  `samples` (2026-10-16, branch `fix-utc-day-boundary`); data rebuilt (27396).
- [x] Final protocol retrained on the rebuilt data (2026-10-16 → 10-17). Every model now uses one
  setting, batch 2048 and max-bins 16; the old runs had mixed settings, e.g. `seq_mta`
  seeds 0–1 at batch 1024, `dcrnn_mta` seed 0 at 4096 and seeds 1–2 at max-bins 48.
  - CPU: baselines 27397, hist_avg 27398, lgbm_no_mta 27399, lgbm_mta 27400,
    lgbm_mta_randsplit 27401, lgbm_mta_ho 27402, rashvand_mlp 27403 / _mtalabel 27404.
  - GPU, all on GPU 0 (GPU 1 still awaits an admin reset), three chains
    `jobs/plans/final_{a,b,c}.txt` (27405–27407, `stgnn.pbs PLAN=`). A: headline
    qsdrop, stgnn_mta, seq_mta, STAEformer, A3T-GCN, T-GCN, ConvLSTM e20, q, sdrop. B: DCRNN,
    stgnn_no_mta. C: Graph WaveNet, seq_no_mta, the cold-start set.
  - At 15:48 root deleted all three chains (4.9 h in, after another GPU user had tried to
    delete one), probably for crowding the shared GPU. 9 of 49 runs were done (qsdrop ×3,
    stgnn_mta ×3, seq_mta s0–1, gwnet s0). The rest run **one job at a time** until GPU 1
    is fixed: `jobs/plans/final_serial.txt` (the same 49 runs, most important first,
    DCRNN seeds last), jobs 27432 and 27433 (48 h each, the second `afterany` the first).
    Running alone, an epoch takes 37 s instead of 58 s. All 49 runs finished in job 27432
    (Oct 17, 22:40; DCRNN about 18 min/epoch alone vs 108 min when shared).
  - Evaluations and analyses rerun on 10-17: final, main, paired, uncertainty (+ under
    missing data), error analysis, randsplit, cold start, robustness and permutation
    (27471, 27472; plus stgnn_mta_q 27478 and gwnet_mta 27479), figures. §4–§5 now hold
    the rebuilt numbers; old-build numbers that were not rerun are marked as such. Old
    numbers stay in git history (branch `restart-multistep-eta`).
- [x] **Main model decided (2026-10-18, user):** `stgnn_mta_q` is the main model,
  `stgnn_mta_qsdrop` the robust variant (§4 finding 12). Paired tests rerun with q first
  (`paired.py`), q's intervals scored under missing data (§5.8), figures switched to q
  (`figures.py MAIN`), deck updated.
- [ ] Next rebuild: MTA availability mask from the matched snapshot's age (code review 1).
- [x] GWNet in the cold-start set (job 27480, `jobs/plans/coldstart_gwnet.txt`, with
  `stgnn_mta_ho_abl` seeds 1–2; done 2026-10-18): it fails like our transductive encoder,
  109.3 s, worse than no graph (§5.4).
- [x] GWNet with the quantile head, 3 seeds (2026-10-18, job 27485): 96.95 s, slightly
  worse than GWNet with L1; the main model beats it by 0.34 s (§4 finding 4).
- [ ] Optional: GWNet trained with observation dropout.
- [ ] Second borough (deferred by the user 2026-10-16; Queens or Brooklyn, whichever is
  better). Checked feasible: the archive has Brooklyn's 57 local/SBS routes (about 1.5×
  Manhattan's pings per day), and parquet.gtfsrt.io `schedules/` stores every borough's
  static feed versions (Brooklyn: Feb 26, Mar 28, Apr 9, Jun 24, Sep 3), with 92–95% RT
  trip-id coverage in each period.
  Brooklyn vs Queens (2026-10-18; RT of Wed Sep 16 and Wed Mar 18, current static feeds;
  `data/raw/boro_compare/boro*.py`). Local + SBS routes, express excluded:

  | | Manhattan | Brooklyn | Queens |
  |---|---|---|---|
  | routes in RT | 42 | 57 | 92 |
  | pings per weekday | 0.79M | 1.22M (1.5×) | 1.57M (2.0×) |
  | trips / vehicles per weekday | 8.7k / 671 | 12.1k / 1,166 | 18.3k / 1,558 |
  | static feeds (share of pings) | `gtfs_m` | `gtfs_b` 55 routes (95%), `gtfs_busco` 2 (B100, B103) | `gtfs_busco` 46 (50%), `gtfs_q` 37 (40%), `gtfs_b` 9 (10%) |
  | median stops per trip | 21 | 41 | 24 |
  | route set Mar → Sep | | stable (B98 dropped) | stable |

  Neither borough changed its network in the window: the Queens redesign took effect in
  2025 (Jun 29 and Aug 31), and Brooklyn's is still at the plan stage. The pipeline reads
  one static feed. Brooklyn's NYCT routes (`gtfs_b`, 95% of its pings) therefore need only a new route prefix and feed URL;
  Queens needs three feeds merged. Stops shared between boroughs: Brooklyn–Queens 80,
  Queens–Manhattan 24, Brooklyn–Manhattan 2. Median ping interval is 30 s everywhere. RT route
  ids are upper case, so the Bronx (`BX…`) must be excluded explicitly when filtering by `B`.

## 9. Reproduction pointers
- Pipeline: `fetch_rt` → `fetch_static` → `static_gtfs` → `arrivals` → `samples` →
  `context` → `graph` → `segment_attrs` → `network_state` → `models/stgnn_data` → models
  → `evaluate`. See README.
- Results CSVs: `data/processed/results/<name>.csv`, `_significance.csv`, `_seeds.csv`.
  Side analyses: `data/processed/preds/analysis/`.
- Job scripts: `jobs/*.pbs`. Logs: `jobs/logs/`.
