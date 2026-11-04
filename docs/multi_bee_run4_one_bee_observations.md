# Multi-bee run 4 (`mb_bee1_s0`): one bee through the multi-bee code

Written 2026-11-01. Source: `kaggle-checkpoints/multibee_checkpoints/mb_bee1_s0/` (`summary.json`, `episodes.csv`, `updates.csv`, `log.txt`, checkpoints 10k-30k, `final/`). Companions: runs 1-3 in `multi_bee_run1/2/3_*.md`.

## 1. Setup

`multi_bee_training.py --bees 1 --steps 30000 --seed 0 --read-start 999999 --buffer 200000`. One worker, shaping off, no rescue. Everything else as in run 3. Runtime 857 s, 35.1 steps/s (about twice the 3-bee rate; the earlier guess that throughput was learner-bound was wrong, a single worker is faster per step).

## 2. Observations

| # | Observation | Evidence |
|---|---|---|
| 1 | **Zero arrivals in 30k steps.** 110 episodes: 60 out-of-road, 50 timeouts, bee 0 NOVICE, E = 0. | `summary.json` |
| 2 | **Same stall, one bee.** The first post-warmup episode (step 5023, 26.1 m) is out-of-road; after that all 50 episodes are timeouts. First timeouts at steps 5115, 5615, 6115 ... with paths 6.0-9.4 m. | `episodes.csv` |
| 3 | **Windows 10-20k and 20-30k are 100% timeouts**, 20 episodes each, max path 9.1 and 7.6 m, mean 7.1 and 6.1 m. | `summary.json` |
| 4 | **Entropy collapses faster than in the single-bee loop.** Entropy 1.37 at 6.1k, 1.21 at 12.1k, 0.76 at 14.1k, 0.11 at 16.1k, -0.59 at 20.1k, -0.18 at 25.1k, 0.86 at 30.1k. Alpha 0.055 at 16.1k. Single-bee at 16k: entropy 0.86, alpha 0.041. | `updates.csv`, `logs/shaping_none_s0.log` |
| 5 | **Single-bee exits at about step 14.5k.** Its stall (episodes 72-90, max speed 3.6-5.8 km/h) ends at episode 91 when max speed rises to 7-10 km/h and episodes end in crashes. No such step appears here. | `logs/shaping_none_s0.log` |

## 3. Conclusion so far

Parallelism (three bees on one policy) is **not** the cause. One bee through `multi_bee_training.py` stalls the same way. Shaping (run 3), the 50k buffer (run 3) and the rescue (run 2) are also ruled out. What is left: something about how `multi_bee_training.py` collects and learns differently from `sandbox_bee2_training.py`, or something about Kaggle versus the local machine.

## 4. Differences in the code that could matter (read from `worker_main`, `run_episode`)

1. **Weights change only between episodes.** The worker applies new weights at the top of its loop and then runs a whole episode with fixed weights (`worker_main`, lines about 170-190). The single-bee loop updates every step.
2. **The learner does not learn while an episode is running.** Transitions are sent only when the episode ends. In a stall that is 500 steps with no learning, then 500 updates in a row. The acting policy is therefore always at least one full episode behind the data.
3. **The first post-warmup episode is acted by untrained weights.** Learner updates start only after 5000 transitions have arrived, so the first policy-driven episode (steps 5000-5500) uses the initial network. Its near-zero actions would match the ~7 m creep. Not checked (actions are not stored in `episodes.csv`).
4. **Platform.** Kaggle: Linux, Python 3.10.22, numpy 2.2.6, torch 2.14.1+cpu. Local: Windows, numpy 2.2.6, torch 2.13.0. The versions are close but not identical.

## 5. Next test

Run the single-bee script (`sandbox_bee2_training.py --shaping none --seed 0`) **on Kaggle**, 30k steps. It exited its stall at about 14.5k when run locally.
- Escapes on Kaggle: the platform is fine and the cause is in the multi-bee collect/learn loop (items 1-3 above).
- Stalls on Kaggle: the problem is the platform or package versions, not the loop.

## 5b. Result of that test (`kg_single_none_s0`, single-bee script on Kaggle, 30k steps, seed 0)

Source: `kaggle-checkpoints/checkpoints/kg_single_none_s0/`. That script saves only `sandbox_bee_checkpoint.pt`, `pheromone_table.pkl` and `run_config.json`; the per-episode lines were printed to the notebook and are **not** in the folder. So this result is **inferred from the pheromone table**, not read from episodes.

Warning (crash) cells by the step they were last written, 5k buckets:

| | 0-5k (warmup) | 5-15k | 15-20k | 20-25k | 25-30k |
|---|---|---|---|---|---|
| Kaggle single-bee, 30k | 39 | **0** | 6 | 4 | 5 |
| Local single-bee, 60k (`shaping_none_s0`) | 37 | **0** | 10 | 2 | 1 |
| Multi-bee runs 1-4 | 33-35, table never grows after about step 5.4k | | | | |

Reading: the single-bee script on Kaggle has no crashes from 5k to 15k (the stall: timeouts write no warnings), then crashes from 15k onward, the same pattern as the local run. The multi-bee runs never get a crash after warmup. `success written: 0` for Kaggle at 30k is not informative: NOVICE arrivals write no success trace and promotion needs about 4 arrivals; the local run's first arrival was at 27.7k.

Conclusion: **the platform is not the problem.** The single-bee loop leaves the stall on Kaggle as it did locally. The difference sits in the multi-bee collect/learn loop (section 4, items 1-3). To confirm with episode lines, paste the `Episode ...` lines from the notebook cell, or rerun with output saved to a file.

## 5c. Code change made after this (2026-11-02): chunked workers, backpressure, lag logging

Found while reading `run_multi_bee` / `worker_main`: **workers are not throttled.** `result_q` is unbounded, the learner is slower than the simulator, so a worker plays far ahead of what the learner has processed. A worker only picks up new weights between whole episodes. Both mean the data is generated by an old policy. Now measurable:

- `--chunk-steps N`: worker sends transitions every N steps and checks for new weights between chunks. Not usable with `--stall-rescue` (the rescue needs the episode summary before the transitions).
- `--max-lag N`: after each chunk a worker waits while more than N transitions are sent but not yet received by the learner. Needs `--chunk-steps`.
- `--sync-every K`: updates between weight broadcasts (default 200, or 25 with chunking).
- Every transitions message carries the learner update count of the weights that produced it. The learner logs `lag=` (mean updates between those weights and the learner's current update count) on each `[Update ...]` line and in a new `mean_lag` column of `updates.csv`. This is the first direct measurement of the policy lag; no earlier run has it.
- Checked only with a stub environment (instant steps, 2 bees, 1500 steps), not with MetaDrive: old behaviour lag 143-700 updates, chunked with `--max-lag 100` about 78-149. The stub exaggerates the old lag (its steps are free); the real values come from the next Kaggle run.
- Defaults are unchanged (`--chunk-steps 0`), so earlier commands behave as before, apart from the extra `lag=` output.

## 6. Not yet known

- What the stalled actor outputs (throttle, steer). A parked car (zero reward after about 7 m) would fit the numbers but is untested.
- A second seed.
