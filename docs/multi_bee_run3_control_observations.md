# Multi-bee run 3 (`mb_none_s0`): no-shaping control, 200k buffer

Written 2026-11-01. Source: `kaggle-checkpoints/multibee_checkpoints/mb_none_s0/` (`summary.json`, `episodes.csv`, `updates.csv`, `log.txt`, checkpoints 10k-60k, `final/`). Companions: `multi_bee_run1_observations.md` (no rescue, shaping on), `multi_bee_run2_rescue_observations.md` (rescue on).

## 1. Setup

Same as run 1 (3 bees, 60k steps, seed 0, warmup 5000, penalty 20, horizon 500) with two changes: **shaping off** (`--read-start 999999`, so `reading_on` = 0 for every episode, `shaping_ratio` max 0.0) and **buffer 200k** (`--buffer 200000`, recorded as `buffer_capacity_used`). Stall rescue off. Runtime 3275 s (18.4 steps/s).

## 2. Observations

| # | Observation | Evidence |
|---|---|---|
| 1 | **Zero arrivals in 60k steps.** 172 episodes: 110 timeouts, 62 out-of-road. Castes all NOVICE, E = 0. | `summary.json` |
| 2 | **The stall starts at step ~5.2k with shaping fully off.** Post-warmup (after step 5300): 109 episodes, 108 timeouts, 1 out-of-road (step 56.8k, 6.0 m). Max path after warmup 9.6 m. | `episodes.csv` |
| 3 | **Every 10k window after 10k is 100% timeouts at about 7 m.** Mean path 6.8-7.1 m, max 8.3-8.9 m, mean raw return 5.4-6.6. | `summary.json` windows |
| 4 | **Warmup matches runs 1 and 2.** Same seeds, same paths (41.6, 78.6, 60.6 m ...). Episode numbering differs because messages arrive in a different order. | log |
| 5 | **Shaping and the 50k buffer are both ruled out as the cause of the stall.** Run 3 removes both and behaves the same. | obs 1-3 |
| 6 | **The critic loss no longer collapses to 0.00.** With the 200k buffer, last-window critic loss still has spikes (window 50-60k: mean 1.06, min 0.03, max 6.3). The exact-zero tail seen in run 2 (from step 55,213) is absent. The final print is flat at 0.12 only because the last window has one row. | `updates.csv` |
| 7 | **Alpha reaches its floor earlier than in run 1.** Floor (0.03) first reached at step 26.3k, versus 37.5k in run 1. Entropy: 1.37 at 7.8k, 1.07 at 13.8k, 0.73 at 15.8k, about -0.45 at 19.8k, then rises to about 1.1 by 40k. Same collapse-then-rise shape as runs 1 and 2. | `updates.csv` |

## 3. Comparison across the three multi-bee runs

| | run 1 `mb_s0` | run 2 `mb_rescue_s0` | run 3 `mb_none_s0` |
|---|---|---|---|
| Shaping | on | on | **off** |
| Stall rescue | off | **on** | off |
| Buffer | 50k | 50k | **200k** |
| Arrivals | 0 | 0 | 0 |
| Post-warmup timeouts | 112 / 113 | 112 / 112 | 108 / 109 |
| Max path per window after 10k | 8.2-9.0 m | 8.1-8.9 m | 8.3-8.9 m |

## 4. Comparison with the single-bee loop (read from `logs/shaping_none_s0.log`)

- **The single-bee run stalls too, in the same way.** Episodes 72-90 are timeouts at 7-8 m with max speed 3.6-5.8 km/h (about step 5.4k to 14.5k).
- **It leaves the stall at episode 90 by crashing.** At episode 91 the max speed rises to 7-10 km/h and episodes start ending in crashes. Learning to drive then continues, with its first arrival at step 27.7k.
- **Entropy differs.** Single-bee entropy stays about 1.27-1.28 until step 12.5k and is still 0.99 at 14.5k; alpha is 0.06 then. In multi-bee runs entropy is about 1.0-1.07 at 12-14k and falls faster, to about 0.3-0.7 by 16k.
- **Critic loss differs.** Single-bee prints 10-42 in the 5.5k-10.5k window. Multi-bee prints 7-25 with lower spikes. Not conclusive (single prints, spiky metric).

## 5. Hypotheses (not verified)

1. **The multi-bee loop loses the exit the single-bee run has.** The single-bee bee leaves the stall at about step 14.5k while its entropy is still about 1.0, by drifting into a slightly faster speed and crashing. The multi-bee policy's entropy is falling faster at the same stage, so it may become deterministic before that happens. Cause unknown.
2. **Acting with stale weights.** A worker finishes a whole 500-step timeout episode before it applies new weights (control queue polled between episodes; needs checking in `worker_main`). The single-bee policy updates every step. In a stall every episode is 500 steps, so the data a worker produces comes from a policy that is up to about 500 updates old. Not checked.
3. **One update per transition arrives in bursts.** The learner gets 500 transitions at once and does 500 updates, then the next worker's 500. Same number of updates per sample, a different order. Effect unknown.
4. **Three workers acting from identical weights remove between-episode variety.** All three copies make the same stall. Same as run 1 hypothesis 2, still untested; run 1 `--bees 1` through the multi-bee code would test it.

## 6. Still not known

- Whether `multi_bee_training.py` with `--bees 1` and no shaping escapes. This is the cleanest next test: it keeps the new loop and removes only the parallelism.
- Whether a second seed behaves differently.
- Whether the worker applies new weights mid-episode (read `worker_main` around line 170-190).
- What the stalled policy actually outputs (steer/throttle).
