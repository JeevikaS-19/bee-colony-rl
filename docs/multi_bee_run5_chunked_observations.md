# Multi-bee run 5 (`mb_chunk1_s0`): one bee, chunked workers with backpressure

Written 2026-11-02. Source: `kaggle-checkpoints/multibee_checkpoints/mb_chunk1_s0/` (`summary.json`, `episodes.csv`, `updates.csv` with the new `mean_lag` column, `log.txt`, checkpoints 10k-30k, `final/`). Companions: runs 1-4 in `multi_bee_run1..4_*.md`. Code change described in run 4 file, section 5c.

## 1. Setup

`multi_bee_training.py --bees 1 --steps 30000 --seed 0 --read-start 999999 --buffer 200000 --chunk-steps 25 --max-lag 100` (sync every 25 updates by default). Shaping off, no rescue. Identical to run 4 (`mb_bee1_s0`) except for the chunking and backpressure. 714 s, 42.0 steps/s. The optional `--max-lag`-free lag measurement (`mb_lag_old`) was not run, so the old mode's real lag is still unmeasured.

## 2. Observations

| # | Observation | Evidence |
|---|---|---|
| 1 | **Zero arrivals in 30k steps.** 112 episodes: 67 out-of-road, 45 timeouts. | `summary.json` |
| 2 | **Policy lag was held at about 146 updates.** `mean_lag` is 146 on almost every print (mean 141, max 146, min 11 at the very start), 130 in the last. That is roughly `max_lag` (100) plus one chunk plus the broadcast interval (25 each). The backpressure works as intended. | `updates.csv` |
| 3 | **The stall still happens.** Steps 5k-25.9k: every post-warmup episode is a timeout. Window 10k-20k: 20 timeouts, max path 5.9 m, mean 3.6 m (shorter than the 7 m creep of runs 1-4). | `summary.json` windows |
| 4 | **It exits the stall, late.** First post-warmup non-timeout is at step 25,923 (out-of-road, 10.5 m). Window 20k-30k: 16 timeouts and 8 out-of-roads, max path 12.8 m. Further exits: 26.3k (10.0 m), 26.4k (8.5 m), 26.7k (9.0 m), 26.9k (10.2 m), 27.0k (8.5 m), 27.3k (12.8 m), 27.7k (12.8 m). | `episodes.csv` |
| 5 | **Entropy still collapses early.** 1.04 at 13k, 0.68 at 15k, 0.23 at 17k, then 0.29-0.57 until about 25k, 1.09 at 30k. Alpha hits the 0.03 floor by about 20k. | `updates.csv` |

## 3. Comparison (all `--bees 1` except the single-bee references)

| | Steps to first post-warmup crash/out-of-road | Notes |
|---|---|---|
| single-bee script, local (`shaping_none_s0`) | about 14.5k | first arrival at 27.7k |
| single-bee script, Kaggle (`kg_single_none_s0`) | about 15k (from the table, inferred) | not confirmed from episode lines |
| multi-bee code, `--bees 1`, old behaviour (`mb_bee1_s0`) | **none in 30k** | 50 timeouts, 1 out-of-road at 5.0k |
| multi-bee code, `--bees 1`, chunked + `--max-lag 100` (`mb_chunk1_s0`) | **25.9k** | still no arrival by 30k |
| multi-bee code, 3 bees, old behaviour (runs 1-3) | none in 60k, except one out-of-road at 56.8k in `mb_none_s0` | |

## 4. Reading

- Chunking plus backpressure **helped but did not fully close the gap**. The one-bee multi-bee loop now leaves the stall at 25.9k, where the old one did not leave it in 30k. The single-bee script leaves at about 14.5k.
- **One seed per cell.** The earlier single-bee runs differed from each other by tens of metres at 30k, so a difference of 11k steps in the exit step could be seed noise. Not a verdict.
- The remaining lag (about 146 updates) is not zero, and the single-bee loop has none. Entropy at the same alpha is still lower than in the single-bee run (0.23 at 17k here versus 0.86 at 16k in the single-bee log).

## 5. Next test (proposed)

Tighten everything the code can tighten: `--chunk-steps 5 --max-lag 10 --sync-every 5` (expected lag about 15-25 updates). If the exit moves toward 14.5k, lag is the cause and the effect is dose-dependent. If it stays at about 25k, lag is not the main factor and the first-episode or entropy differences are the next suspects. A second seed (`--seed 1`) of both the tight run and the single-bee script would show how much of this is noise.

## 6. Not yet known

- The real policy lag of the old loop (`mb_lag_old` was skipped).
- Whether the 25.9k exit is seed noise.
- What the stalled actor outputs; why entropy falls faster than in the single-bee run.
