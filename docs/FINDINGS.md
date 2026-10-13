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
| Service alerts | MTA bus alerts Mar 1–Oct 12. Route-level free text only (cause/effect empty), about 9 Manhattan alerts per day. Downloaded, not yet used |
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

### Splits and protocol (paper section: Experimental setup)
- **Chronological split:**
  - train 2026-03-01 → 07-31;
  - validation 08-01 → 08-31;
  - test 09-01 → 10-11 (42 days; a new schedule pick D6).
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

## 4. Main results (test; `all` = horizons 1–20; mean over seeds)

| Model | Seeds | MAE (s) | RMSE | MAPE % | ≤ 60 s % | Skill vs mta_cal |
|---|---|---|---|---|---|---|
| **stgnn_mta (ours)** | 3 | **96.7 ± 0.1** | **158.8** | 15.2 | 49.3 | **+16.4%** |
| stgnn_no_mta | 3 | 96.9 ± 0.1 | 159.1 | 15.2 | 49.2 | +16.3% |
| gwnet_mta | 2 | 97.1 | 159.8 | 15.2 | 49.3 | +16.0% |
| seq_mta (no graph) | 3 | 98.5 ± 0.2 | 163.2 | 15.4 | 48.9 | +14.9% |
| lgbm_mta (20% data) | 1 | 99.2 | 183.3 | 15.4 | 48.9 | +14.3% |
| seq_no_mta | 3 | 99.9 ± 0.2 | 166.2 | 15.6 | 48.5 | +13.7% |
| lgbm_no_mta | 1 | 100.7 | 185.6 | 15.6 | 48.5 | +13.0% |
| mta_cal | – | 115.7 | 191.4 | 18.4 | 43.4 | 0 |
| mta (published) | – | 117.5 | 197.4 | 18.7 | 43.3 | −1.6% |
| rt | – | 135.0 | 237.6 | 21.6 | 38.2 | −16.7% |
| sched | – | 354.8 | 588.6 | 81.8 | 17.1 | −206.7% |

- The 95% day-bootstrap CI of the MAE is 93.7–99.5 for ours and 112.1–119.3 for
  `mta_cal`. Median bias is −2.0 s for ours and +7.4 s for `mta_cal`. Within 120 s:
  73.5% vs 68.0%.
- Test rows on common support: 25,335,879 over 42 days.

**MAE by horizon band (s):**

| Model | h1 | h2–5 | h6–10 | h11–15 | h16–20 |
|---|---|---|---|---|---|
| ours (stgnn_mta) | 28.6 | 57.1 | 95.6 | 127.1 | 153.4 |
| gwnet_mta | 28.8 | 57.4 | 96.0 | 127.7 | 154.2 |
| seq_mta | 28.9 | 57.9 | 97.2 | 129.6 | 156.8 |
| lgbm_mta | **28.3** | 57.9 | 98.0 | 131.0 | 158.4 |
| mta_cal | 38.6 | 70.2 | 113.1 | 150.5 | 182.8 |

### Findings
1. **All learned models beat MTA's deployed predictions on all 42 test days**
   (Wilcoxon p ≈ 5e-13), in every horizon band. Ours is +16.4% overall, +25.9% at h1
   and +16.1% at h16–20.
2. **The graph helps, and the help grows with horizon.**
   - Graph vs no graph (paired, daily): −1.9 s overall with MTA input and −3.0 s without.
   - At h1: −0.3 / −0.5 s. At h16–20: −3.8 / −5.6 s.
   - Better on 100% of days, p = 5e-13.
   - Seed spread is about 0.1–0.2 s, so the gain is about 10× the noise.
   - Interpretation: congestion further downstream is invisible to the bus itself but
     observed by other buses already on that street.
3. **With the graph, MTA's predictions add nothing as input** (96.7 vs 96.9 s, 0.19 s).
   An open-data model reconstructs and exceeds the operator's information. Without the
   graph, MTA input is worth 1.4 s.
4. **The graph matters more than the encoder design.**
   - Graph WaveNet captures about 75% of our encoder's gain over no-graph.
   - Ours vs GWNet: −0.41 s overall, better on 86% of days (p = 2e-7). The gap grows
     with horizon (−0.14 s at h1, −0.69 s at h16–20).
   - DCRNN results are pending.
5. **Simple trees are as good one stop ahead.** LightGBM is the best model at h1 (28.3 s
   vs ours 28.6 s). Complexity pays off only at longer horizons; a blend at h1 is an
   option.
6. **RMSE: neural models fix the tail; trees do not.** LightGBM's RMSE (183) barely
   beats `mta_cal` (191), while ours is 159.
7. **LightGBM details.**
   - Going from 5% to 20% of training issues improved MAE by about 2 s (101.1 → 99.2).
   - Dropping `stop_id` costs only 0.2 s (5% data), so the model is not memorising stop
     identities.
8. **Training longer (20 epochs) didn't help on test** (96.70 vs 96.61 s), despite about
   0.3 s better validation MAE. It overfits the validation month. 10 epochs is the
   protocol.
9. **Consistency.** Predicted arrival times decrease from one stop to the next on only
   0.19% of consecutive pairs for ours (MTA: 0.36%). No additive / monotone head is
   needed.

## 5. Analyses

### 5.1 Missing network data: robustness (test-time only, no retraining; Jiang et al. 2022)
A random share of (bin, segment) observations is hidden, as if fewer buses reported.

| Hidden | 0% | 25% | 50% | 75% | 90% |
|---|---|---|---|---|---|
| stgnn_mta MAE (s) | 96.5 | 97.0 | 98.3 | 101.0 | 104.5 |

- The no-graph model scores 98.4 s, so the graph is worth having while about half or
  more of the observations remain.
- Below that it is *worse* than having no graph, because it was trained only on
  full-coverage data.
- Planned fix: random observation dropout during training.
- Relevance: agencies with partial AVL coverage.

### 5.2 Permutation importance of the graph model (one input group shuffled on test)

| Group shuffled | MAE (s) | Δ vs 96.5 |
|---|---|---|
| Real-time segment estimate (rt, rt_fresh) | 335.6 | +239 |
| Schedule / distances | 171.8 | +75 |
| **Network state (graph; bins shuffled)** | 129.5 | **+33** |
| Bus state (delays, pace, headway, …) | 107.6 | +11 |
| MTA prediction | 104.0 | +7.4 |
| Time of day | 103.8 | +7.2 |
| Route id | 98.0 | +1.4 |
| Day of week | 97.7 | +1.2 |
| **Weather + calendar** | 96.6 | **+0.1** |

- **Weather adds essentially nothing** once the real-time network state is known. This
  goes against common claims in the literature; worth stating.
- Route identity matters little (+1.4 s), so the model generalises across routes rather
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
- **Random-split leakage:** LightGBM trained on a random issue-level split over all
  months — PENDING (job 27224).

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

Results on the 6 held-out routes (MAE, s):

| Model | All | h1 | h16–20 | Skill vs mta_cal |
|---|---|---|---|---|
| stgnn_mta, trained *with* these routes (upper bound) | 96.5 | 28.9 | 151.9 | +16.5% |
| seq_mta, trained with these routes | 98.2 | 29.2 | 155.3 | +15.1% |
| **seq, held out + ablation** | **103.0** | 30.5 | 162.4 | **+10.9%** |
| stgnn, held out + ablation | 110.6 | 31.2 | 176.2 | +4.3% |
| stgnn, held out, no ablation | 115.3 | 36.1 | 183.6 | +0.3% |
| seq, held out, no ablation | 116.8 | 36.5 | 185.0 | −1.0% |
| mta_cal | 115.6 | 40.7 | 179.9 | 0 |

- **BusTr's ablation is essential:** it gains 7.6–13.8 s on unseen routes.
- **Negative result: our graph encoder hurts on unseen routes** (110.6 vs 103.0 s). Its
  per-segment learned vectors and learned adaptive adjacency memorise segment identity,
  and they are untrained for unseen segments. The ablation removed route and static ids
  but not these encoder identities.
- **Fix under test:** an *inductive* encoder (`--inductive`). Segments are described by
  physical attributes (signals, bus lanes, length) instead of learned vectors, with no
  adaptive adjacency, in the spirit of MAGTTE's inductive learning. PENDING (nn_v2 runs).
- LightGBM held-out (job 27229) and with attributes (27236) — PENDING.

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
- Runs — all PENDING:
  - LightGBM `+school` (27233), `+attrs` (27234), `+both` (27235);
  - nn_v2: `stgnn_mta_v2`, `stgnn_mta_v2noattr`, `seq_mta_v2`, `stgnn_mta_v2ind`;
  - cold start: `*_v2_ho_abl`, `stgnn_mta_v2ind_ho_abl`.
- School natural experiment: test Sep 1–9 (no school) vs Sep 10+ — analysis TODO.

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
- The encoder-design gain is modest (0.4 s over Graph WaveNet). The contribution is the
  setup and graph, not a radically new network.
- The graph is trained on full-coverage data and degrades below about 50% coverage
  (§5.1).
- Bus-lane direction isn't matched (parallel heading only); OSM signal completeness is
  not verified.
- Evaluated on a 25% hash sample of test issues so far (to be 100% for final tables).

## 8. Open items / TODO
- [ ] DCRNN seeds 0–1 (running), GWNet seed 2 (done; add to table).
- [ ] Random-split pitfall result (LightGBM).
- [ ] Cold-start: LightGBM; inductive encoder; segment attributes.
- [ ] nn_v2: school + attributes on the main test.
- [ ] Training with random state dropout (robustness fix).
- [ ] Error analysis: rush vs off-peak, rain, route, SBS vs local, school start, service alerts.
- [ ] Uncertainty (quantile) outputs and calibration.
- [ ] Petersen ConvLSTM baseline.
- [ ] Final protocol: 3 seeds for every model, EVAL_FRAC = 1.0, freeze hyperparameters.
- [ ] Figures: corridor-graph map, MAE vs horizon, robustness curve, permutation bars,
  error by time of day, a worked trip example.

## 9. Reproduction pointers
- Pipeline: `fetch_rt` → `fetch_static` → `static_gtfs` → `arrivals` → `samples` →
  `context` → `graph` → `segment_attrs` → `network_state` → `models/stgnn_data` → models
  → `evaluate`. See README.
- Results CSVs: `data/processed/results/<name>.csv`, `_significance.csv`, `_seeds.csv`.
  Side analyses: `data/processed/preds/analysis/`.
- Job scripts: `jobs/*.pbs`. Logs: `jobs/logs/`.
