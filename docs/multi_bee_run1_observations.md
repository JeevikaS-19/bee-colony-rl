# Multi-bee run 1 (`mb_s0`): observations

Written 2026-11-01. Source files: `kaggle-checkpoints/multibee_checkpoints/mb_s0/` (`episodes.csv`, `updates.csv`, `summary.json`, `run_config.json`, `log.txt`, `final/`, `checkpoint_10k..60k/`). This file separates **what the data shows** from **what is only a hypothesis**. Cite the first part; do not cite the second as a finding.

## 1. Setup actually used

| Item | Value |
|---|---|
| Script | `multi_bee_training.py`, run on Kaggle (CPU, 4 cores, 31 GB RAM) |
| Python | 3.10 venv built with `uv` (Kaggle's Python 3.13 cannot install MetaDrive 0.4.3) |
| Command | `--run-name mb_s0 --bees 3 --steps 60000 --seed 0` |
| Workers | 3 MetaDrive processes, one shared SAC policy, learner on CPU |
| Warmup / reading | warmup 5000 steps (throttle-biased random), reading and shaping on from step 5000 |
| Shaping | `ShapingCalculator` with eta_success 0.5, eta_crash 1.0 (**not** the frozen beta = 2.86 from `shaping.py`) |
| Pheromone decay | half-life 9000 steps (0.15 x 60k) |
| Env | map C, horizon 500, penalties 20.0, one fixed scenario (index 0) |
| SAC | batch 256, lr 3e-4, gamma 0.99, target entropy 0.0, alpha floor 0.03, one update per transition |
| Deviations from the frozen single-bee settings | buffer 50k (frozen 200k); table sent to workers every 10 episodes or on an arrival; 3 workers per policy |
| Runtime | 3382 s, about 17.4 steps/s |

## 2. Observations (what the data shows)

| # | Observation | Evidence |
|---|---|---|
| 1 | **Zero arrivals in 60k steps.** 175 episodes: 109 timeouts, 66 out-of-road, 0 arrive_dest. | `summary.json` |
| 2 | **Warmup phase (steps 0-5k) looks like the baselines.** 62 episodes, all out_of_road except a few timeouts later; paths up to 78.6 m. | `episodes.csv` |
| 3 | **A timeout stall starts the moment the policy takes over.** First timeout at episode 64, step 5214. After step 5k: 113 episodes, 112 of them timeouts (one out_of_road, episode 117, 5.9 m). | `episodes.csv` |
| 4 | **During the stall the bee barely moves.** Timeout episodes: 500 steps, mean path 7.0 m (range 5.3-9.3), mean return +6.5. | `episodes.csv` |
| 5 | **All three bees stall identically.** Episodes after 5k: bee 0 = 36, bee 1 = 39, bee 2 = 38, all timeouts. | `episodes.csv` |
| 6 | **Every 10k window from 10k to 60k is 100% timeouts** (one out_of_road in 30-40k). Max path per window 8.2-9.0 m. | `summary.json` windows |
| 7 | **No bee was ever promoted.** E = 0.0 for all three, caste NOVICE throughout. | `final/tracker.json` |
| 8 | **The success channel was never written.** Pheromone table: 35 cells, 315 entries, all `crash_offroad`; 0 success, 0 crash_vehicle. Table size flat at 35 from step ~5.4k to the end. | `final/pheromone_table.pkl`, `episodes.csv` (`wrote` column: only `crash` or `none`) |
| 9 | **Shaping was tiny.** With reading on (111 episodes): mean |F|/|r| = 0.024, max 0.051. The single-bee pilots saw 0.11-0.13. | `episodes.csv` (`shaping_ratio`) |
| 10 | **Entropy collapsed, then re-expanded.** Mean entropy by 10k window: 1.34 (0-10k), 0.53, 0.05 (20-30k), 0.04 (30-40k), 0.85, 1.27, 1.32 (50-60k). Alpha fell from 0.86 to the 0.03 floor by step ~38k and stayed there. Entropy started rising at ~38k, before the 50k buffer filled. | `updates.csv` |
| 11 | **Losses went to zero.** Critic loss by window: 16.0, 8.5, 5.3, 2.3, 1.6, 0.30, 0.004. Actor loss rose from about -14 to -8. | `updates.csv` |
| 12 | **The late policy is high-entropy and still does not drive.** At steps 50-60k entropy is about 1.3 yet episodes are still 500-step timeouts at 6-8 m. | `updates.csv`, `episodes.csv` |
| 13 | **Warnings are all near spawn.** Table cells span x 6 to 80, y -2 to 6. Crash warnings only come from the first ~5.4k steps. | pickle, `episodes.csv` |

## 3. Comparison with the single-bee references (same code family, different loop)

| | none_s0 | none_s1 | multi-bee mb_s0 |
|---|---|---|---|
| Stall after warmup | yes, about episodes 66-82 | yes, about episodes 76-83 | yes, from episode 64 |
| Stall length | about 17 episodes, then recovers | about 8, later one of about 16 | **113 episodes, never recovers** |
| First arrival | step 27,705 (episode 150) | step 32,553 (episode 147) | **none by 60k** |

The stall itself is not new. Both single-bee references go through one. What differs is that this run never leaves it.

## 4. Hypotheses (not verified)

1. **The stall is a local optimum of the reward, and shaping is not the cause.** Creeping earns about +6.5 per 500 steps and a crash costs -20. Evidence for: it starts at step 5.2k when the table held only crash warnings and |F|/|r| is 2-3%. Evidence against: none yet, but the no-shaping control has not been run.
2. **Shared policy removes exploration diversity.** Three bees copy one actor, so once it collapses all three collapse together. The single-bee run has one stochastic agent per policy and escaped. Not tested.
3. **Late entropy re-expansion is a flat-Q effect.** With every stored transition a stall transition, Q-values are nearly constant, losses go to about 0, and the entropy bonus (alpha at its floor) pushes the policy toward uniform. A near-uniform policy barely moves the car, so it still times out. Not tested.
4. **The reputation gate makes the success channel hard to reach.** Novices write no success trace and a bee needs E >= 0.3 (about 4 recent arrivals) to be promoted. With no arrivals at all this was never the binding constraint in this run. It would matter once arrivals appear.
5. **Buffer size (50k) is probably not the cause.** Entropy re-rose from about 38k, before the buffer wrapped at 50k. Possible minor contribution after 50k.

## 5. Not yet known

- Whether the same multi-bee loop **without shaping** (`--read-start 999999`) also stalls. This is the key control.
- Whether a second seed behaves differently (the single-bee references differed by 54 m at 30k).
- Whether the multi-bee loop differs from `sandbox_bee2_training.py` / `single_bee_training.py` in some detail that matters (worker weight staleness, update-to-data ratio, warmup handling).
- What the final policy actually does (the checkpoints are saved, no render check done).

## 6. Reproduce

```bash
# Kaggle: Internet on; uv venv --python 3.10; uv pip install metadrive-simulator==0.4.3 numpy torch (cpu index)
python -u multi_bee_training.py --run-name mb_s0 --bees 3 --steps 60000 --seed 0 --out-root /kaggle/working/multibee_checkpoints
# control (planned): add --read-start 999999
```

## 7. Stall rescue ("Layer 1") as implemented, and where it differs from the spec

Code: `stall_rescue.py` (11 self-tests, `python stall_rescue.py`), wired into `multi_bee_training.py` behind `--stall-rescue` (off by default, so plain runs are unchanged). Learner-side: workers report the start state and actions, the learner adds the bonuses to the transition rewards before they enter the buffer.

| Spec item | What was done | Why |
|---|---|---|
| Stall = zero arrivals in the last 20 episodes | Default: last 20 **post-warmup** episodes all **timeouts**. `--stall-mode arrival` gives the literal rule (still post-warmup). | Replayed on `mb_s0`: the literal rule fires at episode 20, step 1448, inside warmup. It is true in every healthy run (baselines first arrive at episode 147-150). The timeout rule fires at episode 86, step 15,034 (arrival rule, post-warmup: episode 82, step 13,034). |
| Cosine similarity of actions | L2-based similarity, `1 - dist / 2.83`, threshold 0.90 | Stall actions are near-zero vectors; cosine ignores magnitude. |
| Diversity scale 0.2 | Default 0.005 | Measured on a creeping episode with three bees copying: 0.2 gives about -200 per episode, 0.02 about -20 (worse than a crash, about -17), 0.005 about -5. Creeping return is about +6.5. |
| `dict.popitem()` to drop the oldest | `OrderedDict.popitem(last=False)` | `dict.popitem()` removes the newest item. |
| Placed in `shaping_calculator.py` | Separate learner-side module | The diversity and exploration terms need other bees' actions; only the learner sees them. `shaping_calculator.py` is shared with the single-bee loop. |
| State key (position, heading, speed) | Discretised: 2 m cell, 8 heading bins, 5 km/h speed bins, same for all bees | Raw floats never match across bees. |
| Per-step units about +-1 | Total clipped to +-1 per step | Guard against the critic blowing up. |

Kept as specified: the flag latches (never unflagged); history and visit counts are per bee; action stored against the state **before** the step; comparison only with other bees' **past** episodes; all terms are zero until the flag is on.

Things to remember when writing up:
- These terms are **not potential-based**, so they change the optimal policy. This is a different experiment from the plain shaping arm.
- Any rescue arm needs its own control: same flags plus `--read-start 999999`.
- The flag is latched, so the policy keeps being trained with a bonus that will not exist at solo test time.
- Forward distance is already rewarded by MetaDrive's own driving reward (about 0.9 per metre), so the extra forward bonus raises that rate only by about half (0.5 per metre). It does not change the -20 crash cost.
- Not tested against real MetaDrive yet (local RAM was too low). Unit tests and a replay of the `mb_s0` episodes only.
