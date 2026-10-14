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
| gwnet_mta | 3 | 97.1 ± 0.0 | 159.6 | 15.2 | 49.2 | +16.0% |
| dcrnn_mta | 2 | 97.5 ± 0.2 | 160.8 | 15.3 | 49.1 | +15.7% |
| convlstm_mta (Petersen et al. 2019) | 1 | 100.0 | 162.5 | – | – | +13.6% |
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
| gwnet_mta | 28.8 | 57.4 | 96.0 | 127.7 | 154.1 |
| dcrnn_mta | 28.9 | 57.5 | 96.2 | 128.3 | 155.0 |
| convlstm_mta | 29.3 | 58.6 | 98.5 | 131.8 | 159.8 |
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
   - DCRNN (2 seeds) is 97.5 ± 0.2 s: ours is better by 1.1 s on 98% of days
     (p = 9e-13, seed 0 vs seed 0). DCRNN also costs 3–10× more GPU time per epoch.
   - Ranking ours < GWNet < DCRNN < no graph holds in every horizon band.
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
10. **Petersen et al.'s ConvLSTM** (`--arch convlstm`, a faithful re-implementation
    given the same inputs). It runs a ConvLSTM over the (time × downstream link) grid of
    the bus's own route, then outputs every horizon at once.
    - 100.0 s, +13.6% vs MTA (100% of days). That reproduces the paper's own claim that
      it beats the operator.
    - It is worse than our no-graph sequence model (+1.5 s, 93% of days) and than
      LightGBM on MAE (+0.7 s), but has better RMSE than LightGBM (162.5 vs 183.3).
    - Ours beats it by 3.4 s on 100% of days (p = 5e-13). The gap grows from 0.7 s at
      h1 to 6.5 s at h16–20. Route-local convolution sees only the bus's own corridor;
      the graph sees crossing and parallel routes.
    - Validation MAE was still falling at epoch 10 (90.7 vs our 88.0). A 20-epoch run is
      queued to rule out under-training.

## 5. Analyses

### 5.1 Missing network data: robustness (test-time only, no retraining; Jiang et al. 2022)
A random share of (bin, segment) observations is hidden, as if fewer buses reported.

| Hidden | 0% | 25% | 50% | 75% | 90% |
|---|---|---|---|---|---|
| stgnn_mta MAE (s) | 96.5 | 97.0 | 98.3 | 101.0 | 104.5 |
| **stgnn_mta_sdrop** (trained with observation dropout) | 96.7 | 96.8 | 97.1 | 97.5 | **98.4** |

- The no-graph model scores 98.4 s, so the graph is worth having while about half or
  more of the observations remain.
- Below that it is *worse* than having no graph, because it was trained only on
  full-coverage data.
- **Fix: observation dropout during training** (`--train-state-drop 0.9`). Each
  training bin hides a random share p ~ U(0, 0.9) of its (step, segment) observations.
  - Degradation at 90% hidden falls from +8.0 s to +1.7 s. The model now matches the
    no-graph model (98.4) even with only 10% of observations, and beats it at every
    coverage ≥ 25%.
  - Cost at full coverage: +0.14 s (96.66 vs 96.52, 29% of days better, p = 0.003),
    about one seed s.d.
  - It also leans less on the graph: shuffling the network state costs +17.6 s (vs +33
    without dropout). It uses the state but no longer depends on it.
  - Recommendation for the paper: report both. Use dropout for deployment, or for any
    agency with incomplete AVL coverage.
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
  months — PENDING (resubmitted as job 27249: the first run stalled on an overloaded node).

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
- **Results on the main test (seed 0, paired daily, 42 days):**

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
  - `+both` (27248): still running.
  - Statistically detectable, practically negligible; the same conclusion as for the
    neural models.

### 5.6 Error analysis by operating condition (`python -m bus_eta.error_analysis`)
MAE (s) on the main common support, all horizons. Skill is vs `mta_cal`; graph gain is
seq_mta − stgnn_mta. Files: `results/error_analysis.csv`, `error_analysis_by_route.csv`.

| Condition | Rows | mta_cal | MTA published | lgbm | seq | ours | Ours skill | Graph gain |
|---|---|---|---|---|---|---|---|---|
| Night 0–6 | 2.2M | 83.0 | 82.8 | 66.0 | 66.5 | 65.7 | 20.8% | 0.8 |
| AM peak 6–10 | 5.4M | 107.0 | 113.6 | 89.6 | 89.5 | 88.0 | 17.8% | 1.5 |
| Midday 10–16 | 8.2M | 131.4 | 134.3 | 114.4 | 113.6 | 111.0 | 15.5% | 2.6 |
| PM peak 16–20 | 5.5M | 122.2 | 121.8 | 105.3 | 105.1 | 103.3 | 15.5% | 1.8 |
| Evening 20–24 | 4.1M | 104.4 | 101.9 | 91.1 | 88.1 | 86.8 | 16.8% | 1.3 |
| Weekday | 19.2M | 118.1 | 119.8 | 100.8 | 100.3 | 98.7 | 16.5% | 1.7 |
| Weekend | 6.2M | 108.1 | 110.4 | 94.2 | 92.5 | 90.2 | 16.5% | 2.2 |
| Dry | 20.7M | 117.1 | 119.0 | 100.5 | 99.7 | 98.0 | 16.3% | 1.7 |
| Light rain (<1 mm/h) | 3.1M | 111.7 | 113.2 | 95.7 | 94.7 | 92.1 | 17.5% | 2.6 |
| Heavy rain (≥1 mm/h) | 1.5M | 104.6 | 106.1 | 88.8 | 88.3 | 86.3 | 17.4% | 1.9 |
| SBS routes | 2.6M | 106.0 | 105.7 | 88.5 | 88.2 | 86.8 | 18.1% | 1.4 |
| Local routes | 22.7M | 116.8 | 118.9 | 100.4 | 99.6 | 97.7 | 16.3% | 1.9 |
| Service alert live on route | 16.7M | 119.9 | 122.2 | 103.4 | 102.4 | 100.4 | 16.3% | 2.0 |
| No alert | 8.6M | 107.4 | 108.5 | 91.0 | 90.7 | 89.2 | 17.0% | 1.5 |

- **The gain is uniform.** Skill vs MTA is 15.5–20.8% in every time period, day type,
  weather and route type. **All 41 routes improve** (skill 9.9–30.9%). The graph gain
  is ≥ 0 on every route (0.0 on M35, up to 4.5 s on M20, 4.2 on M50, 3.8 on M12).
- **Midday, not the peaks, is hardest** for every predictor (ours 111 s vs 88 / 103 s at
  the AM / PM peaks). It is also where the graph helps most (2.6 s): long dwell times,
  deliveries and curb conflicts make downstream conditions matter more.
- **MTA's published ETA is worst in the AM peak** (−6.1% vs the calibrated version):
  its bias is time-dependent. Our models need no such correction.
- **Rain does not make prediction harder.** Rainy hours have *lower* error for all
  models; this is confounded with time of day, so it is not causal. It agrees with the
  near-zero weather permutation importance (§5.2): the network state already reflects
  the weather's effect on traffic.
- **SBS routes** (off-board fare, bus lanes) are easier (86.8 vs 97.7 s) and have higher
  skill.
- **Service alerts** are live on 66% of route-issues. The MTA texts are generic ("You may
  wait longer…"). Error is about 11 s higher when an alert is live, for every model.
  Skill drops only slightly (16.3 vs 17.0%). Alerts mark harder conditions but carry
  little specific information; a free-text alert feature is low priority.
- The school-start split is in §5.5.

### 5.7 Uncertainty: quantile head (`--quantiles 0.05 0.1 0.25 0.5 0.75 0.9 0.95`)
Same graph model, one output per quantile, pinball loss; the median is the point ETA.
Scored with `python -m bus_eta.uncertainty` (`results/uncertainty.csv`).

- **The point accuracy is not hurt.** The median scores MAE 96.40 vs 96.52 for the L1
  model (−0.12 s, 67% of days, p = 0.01; within seed noise). Validation MAE 87.9 is the
  best of any run.
- **Calibration** (all horizons): share of arrivals below q5 / q10 / q25 / q50 / q75 /
  q90 / q95 is 4.8 / 9.6 / 24.2 / 48.6 / 73.6 / 89.0 / 94.2%. Quantiles never cross.

| Horizon | 50% int. coverage | width (s) | 80% int. coverage | width (s) | 90% int. coverage | width (s) |
|---|---|---|---|---|---|---|
| h1 | 50.6 | 46 | 80.6 | 89 | 90.5 | 118 |
| h2–5 | 49.8 | 90 | 79.8 | 175 | 89.8 | 231 |
| h6–10 | 49.4 | 150 | 79.3 | 291 | 89.3 | 380 |
| h11–15 | 49.2 | 199 | 79.0 | 385 | 89.1 | 503 |
| h16–20 | 48.9 | 239 | 78.8 | 463 | 88.9 | 603 |

- Slight under-coverage at long horizons (−1.2 pt at h16–20 for 80%) is a typical
  test-month shift. It could be corrected with split-conformal recalibration on
  validation (future work).
- **The intervals are adaptive, not just horizon-wide bands.** Split test predictions
  into fifths by 80% interval width: mean absolute error is 33 / 61 / 87 / 116 / 184 s,
  while coverage stays 80.3 / 79.5 / 79.2 / 79.2 / 78.7%. Spearman(width, |error|) =
  0.51. The model knows which ETAs are unreliable.
- Use case: "arriving in 6–9 min" displays, and transfer / missed-connection risk.
  MTA publishes only a point ETA.

### 5.8 Candidate final model: quantile head + observation dropout (`stgnn_mta_qsdrop`)

| Hidden network observations | 0% | 25% | 50% | 75% | 90% |
|---|---|---|---|---|---|
| Median MAE (s) | 96.6 | 96.8 | 97.0 | 97.5 | 98.3 |
| 90% interval coverage (%) | 89.2 | – | 89.3 | – | 89.3 |
| 90% interval mean width (s) | 383 | – | 385 | – | 388 |

- The two improvements combine without conflict. Accuracy is within seed noise of the
  best model at full coverage (96.4–96.7). It is robust to missing data like
  `stgnn_mta_sdrop`, and calibrated like `stgnn_mta_q`.
- **The intervals stay calibrated when data go missing.** Widths grow slightly (+1%)
  instead of coverage collapsing.
- Shuffling the network state costs +21.7 s (vs +33 for the plain model).
- Proposal: make this the paper's main model. Train it for 3 seeds in the final
  protocol, alongside the inductive variant for cold start.

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
- [x] DCRNN seeds 0–1, GWNet seed 2 (in table).
- [ ] Random-split pitfall result (LightGBM).
- [x] Cold-start: inductive encoder; segment attributes.
- [x] Cold-start: inductive controls (len / none); seeds 1–2.
- [x] Cold-start: LightGBM held out (with and without attributes).
- [x] nn_v2: school + attributes on the main test (no gain).
- [ ] Decide paper framing of the inductive encoder (main model vs cold-start variant).
- [x] Training with random state dropout (robustness fix, §5.1).
- [x] Error analysis: rush vs off-peak, rain, route, SBS vs local, school start, service alerts (§5.6).
- [x] Uncertainty (quantile) outputs and calibration (§5.7).
- [ ] Optional: conformal recalibration of the quantiles on validation.
- [x] Petersen ConvLSTM baseline (finding 10). 20-epoch check queued.
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
