# Multi-bee run 2 (`mb_rescue_s0`, stall rescue ON): observations

Written 2026-11-01. Source: the notebook cell output pasted from Kaggle (`log.txt` content). **The run folder itself is lost**: the Kaggle session reset and `/kaggle/working` was wiped, so there are no CSVs, checkpoints or `summary.json` for this run. Everything below comes from the printed log. The full raw log is not saved in the repo; only the lines quoted here are.

Companion file: `docs/multi_bee_run1_observations.md` (run 1, no rescue). Facts below are separated from hypotheses, as in that file.

## 1. Setup

Same as run 1 except for the flags: `--run-name mb_rescue_s0 --bees 3 --steps 60000 --seed 0 --stall-rescue` (defaults: window 20, mode `timeout`, forward 0.5, diversity 0.005, explore 0.05, similarity 0.90). Python 3.10 uv venv, Kaggle CPU, 3473 s (17.3 steps/s). Warmup 5000, reading on from 5000, buffer 50k, shaping as in run 1.

## 2. Observations (what the log shows)

| # | Observation | Evidence in the log |
|---|---|---|
| 1 | **Zero arrivals in 60k steps.** 175 episodes: 112 timeouts, 63 out-of-road. All 112 post-warmup episodes are timeouts. | final summary |
| 2 | **Warmup is identical to run 1.** Same seeds, same long random paths (ep 31 = 78.6 m, ep 60 = 71.7 m, ep 37 = 60.6 m). All 63 warmup episodes are out-of-road. These long paths are random driving, not learning. | eps 1-63 |
| 3 | **The stall starts the moment the policy takes over.** First timeouts are eps 64-66, step ~5213, paths 5.8-7.7 m, read=1. Same as run 1. | eps 64-67 |
| 4 | **The flag fired at step 14,213 (episode 83).** Run 1 replay predicted ~15,034; the stall began a little earlier here. `[stall check]` at ep 80 showed 19/20, at ep 90 `active=True`. | `*** STALL FLAGGED` line |
| 5 | **The bonuses were applied, and they are small.** Per-episode totals after the flag (9 checks, eps 90-170): forward 2.6-4.4 (mean about 3.5), diversity 0.20-0.36 (mean about 0.27), explore 0.01-0.07 (mean about 0.03). Largest single-step bonus 0.04-0.07, far below the clip of 1.0. Code path: `multi_bee_training.py:353` adds the bonus to the transition reward before `buffer.push`; the printed `R` is the worker's raw env return, so it does not move. | `[stall check]` lines; code |
| 6 | **Forward bonus is just 0.5 x path.** 6-8 m of creeping gives about +3-4 per episode. It pays a creeping bee more, by roughly 50% on top of the +6.5 creeping return. | ep 90: path 6.2 m, fwd 3.12 |
| 7 | **No behavioural change after the flag.** 93 post-flag episodes (83-175), all timeouts, paths 5.3-8.9 m. Pre-flag stall range 5.5-9.6 m. 10k-window max paths: 8.8, 8.2, 8.9, 8.4, 8.1 m (run 1: 8.2-9.0). | episodes, windows |
| 8 | **No bee promoted, no success trace.** All NOVICE, E = 0.00 to the end. Table stayed at 33 cells (all crash warnings from warmup). | `cells=33` every line |
| 9 | **Entropy collapsed, then re-expanded, same shape as run 1.** About 1.35 at 6k, 0.7 at 14k (flag), about 0 from 17k to 28k (some negative readings, -0.4 at 25.7k), then rising to about 1.2-1.3 from about 40k. Alpha hit the 0.03 floor at about step 32.7k (run 1: about 38k). | `[Update ...]` lines |
| 10 | **Critic loss alternates, then goes to exactly 0.00.** From about 8k to 55k the printed critic loss alternates between about 0.1-0.5 and about 4-17 (about one print in three is a spike). From **step 55,213 onward every print is 0.00**. | `[Update ...]` lines |
| 11 | **The 0.00 starts at the step the last crash transition leaves the 50k buffer.** The last out-of-road episode ended at about step 5,213. A 50k buffer drops it when about 55.2k transitions have been pushed. The two numbers match. After that point the buffer holds only stall transitions: no crash, no fast driving. | log arithmetic |
| 12 | **Actor loss drifts from about -14.9 to about -7.9** between 17k and 60k (alpha is 0.03-0.05, so this is almost entirely the critic's Q for the actor's own actions). | `[Update ...]` lines |

## 3. Comparison with run 1 (`mb_s0`, no rescue)

| | mb_s0 | mb_rescue_s0 |
|---|---|---|
| Arrivals | 0 | 0 |
| Episodes (out-of-road / timeout) | 175 (66 / 109) | 175 (63 / 112) |
| Post-warmup timeouts | 112 of 113 | 112 of 112 |
| Max path per 10k window after 10k | 8.2-9.0 m | 8.1-8.9 m |
| Castes at end | all NOVICE | all NOVICE |
| Alpha reaches floor | about step 38k | about step 32.7k |
| Final critic loss | 0.004 | 0.00 |
| Speed | 17.4 steps/s | 17.3 steps/s |

Verdict: **no measurable effect of the rescue.** The small differences (3 episodes between categories, alpha floor 5k steps earlier) are within what one seed can do by itself. The single-bee references differed from each other by much more.

## 4. Hypotheses (not verified)

1. **The bonuses are too small to change the ranking of behaviours.** About +3.5 per episode on a +6.5 base is not a different policy. The forward bonus also pays for creeping. Evidence for: observations 5-7. Not yet tested: a larger forward scale.
2. **The bonus can only reward what the bees already do.** Bonuses are added only to new transitions after the flag, and every new transition is a creep. The warmup transitions that show far driving carry no bonus. The rescue therefore has no way to make "drive far" look better than "creep" in the data. Reasoned from the code, not measured.
3. **The exploration bonus is negligible by construction.** `explore_scale / (visits + 1)` with a coarse state key and about 33k stall steps in few cells gives 0.01-0.07 per episode. Same for diversity. Not tested.
4. **The flag came late.** By step 14k entropy was already collapsing (0.7 and falling), so the policy was nearly deterministic by the time bonuses started. A stall flag at about step 8k would have fired during the high-entropy phase. Untested.
5. **The tail of the run (after about 55k) is an artifact of the 50k buffer.** With every crash transition evicted, the critic only ever sees stall data, so loss 0.00 and rising entropy are expected from flat Q-values, and say nothing about whether the stall is stable. A 200k buffer (the frozen single-bee setting) would have kept the warmup crashes. Does not explain the stall onset at 5.2k.
6. **Open anomaly: the critic's value for the actor's actions looks too high.** Actor loss about -14.9 means Q is about 15 for the actor's own actions, yet creeping pays about 0.013 per step (discounted about 1.3 to 2). Something in the critic expects far more than the creeping data supports; possibly an extrapolation error from the warmup driving data (about 0.3 per step). This is the one finding that does not fit the simple "creeping is the best option in the data" story. Not investigated.

## 5. Still not known

- The no-shaping control (`--read-start 999999`, no rescue). Still the key control for both runs.
- A second seed.
- Whether the multi-bee loop differs from the single-bee loop in a way that matters (update-to-data ratio, weight staleness, three bees on one policy).
- Whether a stronger rescue (bigger forward scale, earlier flag) does anything.
- What the stalled policy actually does with its actions (checkpoints are lost; no render check).

## 6. Housekeeping

- Run files lost with the Kaggle session. If this run is repeated: zip in the same cell right after training, download immediately, or use Save Version.
- The `stall_rescue.py` code and its 11 self-tests are unchanged and unaffected by this result: the code ran as designed (flag at 14,213, bonuses applied, within the clip). The negative result is about the design, not a bug.
