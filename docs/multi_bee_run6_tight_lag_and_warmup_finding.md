# Run 6: tight lag, single-bee seed 1 on Kaggle, and a warmup difference between the two loops

Written 2026-11-02. Sources: `kaggle-checkpoints/checkpoints/mb_tight_s1.log`, `kg_single_none_s1.log`, `mb_tight_s0.log` (only printed logs, no CSVs or checkpoints were brought back), plus `logs/shaping_none_s0.log`, `shaping_none_s1.log` (local single-bee), and the `episodes.csv` of runs 1-5. Companions: runs 1-5 in `multi_bee_run1..5_*.md`.

## 1. What the three new files contain

| File | Content |
|---|---|
| `mb_tight_s0.log` | One line: `.../mb_tight_s0 is not empty. Pick a new --run-name.` **Run A did not run as launched.** The folder already existed (an earlier launch of the same cell was probably still running) and the second launch's `tee` overwrote the first one's log. The output of the first launch was not saved here. |
| `mb_tight_s1.log` | Complete run, `--bees 1 --seed 1 --chunk-steps 5 --max-lag 10 --sync-every 5`, 30k steps. |
| `kg_single_none_s1.log` | `sandbox_bee2_training.py --shaping none --seed 1` on Kaggle. **The log stops at step 18,829** (episode 124, no summary): the run was interrupted before 30k. |

## 2. Observations

| # | Observation | Evidence |
|---|---|---|
| 1 | **Lag was cut to 16 updates** (mean 15.7, max 16) with `--chunk-steps 5 --max-lag 10 --sync-every 5`. Run 5 had 146. | `mb_tight_s1.log` |
| 2 | **At lag 16 the bee still never leaves the stall.** 119 episodes: 70 out-of-road (all in warmup), 49 timeouts; windows 10-20k and 20-30k are 100% timeouts, max path 4.3 m and 7.0 m. Entropy 0.79 at 14k, 0.26 at 18k, -0.07 at 20k, back to 1.09 at 25k and 1.22 at 30k with alpha at the 0.03 floor. 40.2 steps/s. | `mb_tight_s1.log` |
| 3 | **The single-bee script on Kaggle reproduces the local run.** Seed 1: first policy crash after the stall at step 9,366 (Kaggle) vs 9,374 (local `shaping_none_s1`); next crashes at 9,704 vs 9,698 and 10,146 vs 10,133. Same seed gives nearly the same trajectory, so Kaggle and the local machine behave the same. | `kg_single_none_s1.log`, `logs/shaping_none_s1.log` |
| 4 | **Single-bee seed 1 stalls only about 4k steps** (first policy timeout 5,510, first crash after it 9,366) and then crashes in most episodes (steps 5-10k: 8 timeouts, 3 crashes; 10-15k: 4 timeouts, 22 crashes; 15-18.8k: 2 timeouts, 10 crashes), max distance 34 m. No arrival before the log stops. | `kg_single_none_s1.log` |
| 5 | **Lag is not the main cause.** Lag 146 (seed 0): exit at 25.9k. Lag 16 (seed 1): no exit at 30k. Single-bee: exit at 14.5k (seed 0) and 9.4k (seed 1). The one exit at lag 146 is probably noise. | runs 4-6 |
| 6 | **The two SAC implementations are the same algorithm.** `single_bee_training.SACAgent` (used by the multi-bee loop) and the one inside `sandbox_bee2_training.py` have identical losses, alpha update, floor clamp (0.03), Polyak update and sampling; only the code layout and buffer container differ. | read side by side |
| 7 | **The warmup actions differ.** Single-bee: steering `uniform(-0.3, 0.3)`, throttle `uniform(0.2, 1.0)` (`sandbox_bee2_training.py` lines 307-308). Multi-bee: steering `uniform(-1, 1)`, throttle `uniform(0.3, 1.0)` (`multi_bee_training.py` line 114, present since commit `f535c02`). | code |
| 8 | **The warmup data look very different.** Warmup episodes ending in a crash (path in metres): | see table below |

| Warmup episodes | n | mean path | median | share under 6 m | share over 40 m |
|---|---|---|---|---|---|
| single-bee local seed 0 | 71 | 38.9 | 34.1 | 3% | 45% |
| single-bee local / Kaggle seed 1 | 74 | 34.6 | 30.9 | 0% | 39% |
| multi-bee `mb_s0` | 62 | 20.2 | 15.2 | 26% | 15% |
| multi-bee `mb_none_s0` | 58 | 22.5 | 16.9 | 28% | 19% |
| multi-bee `mb_bee1_s0` | 59 | 20.6 | 15.1 | 29% | 17% |
| multi-bee `mb_chunk1_s0` | 58 | 20.8 | 15.2 | 29% | 17% |

About a quarter to a third of multi-bee warmup episodes crash within 6 m (full-lock steering at speed), against 0-3% in the single-bee loop. Every multi-bee run so far uses the same warmup, and every one stalls.

## 3. Hypothesis (not verified)

The wide-steering warmup fills the replay buffer with short violent crashes. The critic sees that nearly every non-trivial throttle ends in -20, while straight driving (which the single-bee warmup produces in quantity: median 31-34 m, each metre worth about 1.0) is thin in the data. After warmup the policy therefore prefers creeping. This fits all of the runs 1-6 evidence (no effect of shaping, buffer, rescue, bee count, or lag), and it is the largest remaining difference between the loops. Not proven: it is a difference in the data, not yet shown to be the cause.

## 4. Code change made after this

`multi_bee_training.py` gained `--warmup-steer` (default 1.0) and `--warmup-throttle-min` (default 0.3). Defaults reproduce the old behaviour. `--warmup-steer 0.3 --warmup-throttle-min 0.2` makes the warmup match the single-bee loop. Both values are recorded in `run_config.json`. Checked only with the stub environment.

## 5. Next test

`--bees 1`, old loop (no chunking), shaping off, warmup 0.3 / 0.2, seeds 0 and 1, 30k steps. Single-bee exits at 14.5k (seed 0) and 9.4k (seed 1). If these runs exit close to those, the warmup is the cause. If they still stall, the difference is elsewhere (first-episode untrained actor, `counter`-based warmup gate, seeding) and the next step is to run the multi-bee loop and the single-bee loop with their random streams matched.

## 5b. Result of the warmup test, seed 1 (`mb_warm_s1`, old loop, narrow warmup) and what it exposed

Source: `kaggle-checkpoints/checkpoints/mb_warm_s1.log` (printed log only). Command: `--bees 1 --seed 1 --steps 30000 --read-start 999999 --buffer 200000 --warmup-steer 0.3 --warmup-throttle-min 0.2`, **no chunking** (this was my mistake: the test did not combine the warmup change with the lag fix).

| # | Observation |
|---|---|
| 1 | **The warmup now matches the single-bee loop.** 75 warmup episodes, mean path 35.5 m, median 29.7 m, 1% under 6 m, 40% over 40 m (single-bee seed 1: 34.6 / 30.9 / 0% / 39%). |
| 2 | **The stall is still there and does not end.** After step 5k: 50 timeouts and 2 out-of-roads (steps 5,051 and 5,590, both before the stall settled). Max path per 5k window: 12.0, 9.7, 8.9, 7.7, 8.5 m. No arrival. Single-bee seed 1 exits at 9.4k. |
| 3 | **The narrow warmup alone does not fix it.** The warmup hypothesis (section 3) is not supported as the sole cause. |
| 4 | **The policy lag in the old loop grows without limit.** `lag=` is 760 at step 6.3k, 2,917 at 8.8k, 5,317 at 11.8k, 10,317 at 17.8k, 20,517 at 29.8k. This is the first real measurement of the old loop (the stub only suggested it). 28.4 steps/s. |

Reading of observation 4: the worker generates data much faster than the learner consumes it and nothing throttles it. The data the learner trains on at update number T was produced by weights from around update T minus the lag, so in this run the data at update 25k came from a policy that had only had about 4-5k updates. In all earlier old-loop runs (runs 1-4, `mb_bee1_s0`, `mb_none_s0`, `mb_s0`, `mb_rescue_s0`) the same thing must have happened, though lag was not logged then. A stall seen in the data could then simply be the early, barely trained policy being replayed to the learner, regardless of what the learner's own policy was doing. This is an inference from the lag numbers, not directly shown.

The two fixes have never been tested together:
- lag fixed, wide warmup: `mb_chunk1_s0` (lag 146, seed 0, exit at 25.9k), `mb_tight_s1` (lag 16, seed 1, no exit by 30k).
- narrow warmup, lag not fixed: `mb_warm_s1` (this run, no exit by 30k).

## 5c. Next test (corrected)

Both fixes at once: `--bees 1 --chunk-steps 5 --max-lag 10 --sync-every 5 --warmup-steer 0.3 --warmup-throttle-min 0.2 --read-start 999999 --buffer 200000`, seeds 1 and 0, 30k steps. Single-bee exits at 9.4k (seed 1) and 14.5k (seed 0). Outcomes: both exit near those: the two differences together explain the stall. Still stuck: something else differs (the first untrained episode, the random streams, the worker-side policy).

## 6. Not yet known

- Whether Run A (tight lag, seed 0) completed on Kaggle (a folder `mb_tight_s0` exists there, its files were not brought back).
- Where the interrupted Kaggle single-bee seed 1 run would have ended (first arrival).
