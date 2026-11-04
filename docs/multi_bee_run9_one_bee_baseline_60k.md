# Run 9: 1-bee baseline to 60k, same flags as the 3-bee runs (`mb1_s0`, `mb1_s1`)

Written 2026-11-04. Sources: `kaggle-checkpoints/results_1bee/` (full CSVs, summaries, checkpoints, logs).
Flags: `--bees 1 --steps 60000 --chunk-steps 5 --max-lag 10 --sync-every 5 --warmup-steer 0.3 --warmup-throttle-min 0.2 --buffer 200000 --read-start 999999`, seeds 0 and 1. Lag 17.6/17.5 mean (max 20/19).

| Run | First arrival | Arrivals | Last-window (50-60k) arrival rate | steps/s |
|---|---|---|---|---|
| 1 bee s0 | 29,119 | 77 | 1.00 | 49.0 |
| 1 bee s1 | 37,817 | 69 | 0.92 | 49.9 |
| 3-bee control s0 | 27,096 | 73 | 0.89 | 21.1 |
| 3-bee control s1 | none | 0 | 0 | 27.7 |
| 3-bee treatment s0 | 32,755 | 67 | 0.78 | 23.6 |
| 3-bee treatment s1 | 39,210 | 47 | 0.97 | 27.8 |

## Reading
1. **No sample-efficiency gain from 3 bees, and no pheromone effect, at this noise level.** 1 bee arrives at 29.1k / 37.8k; 3-bee runs at 27.1k / none (control) and 32.8k / 39.2k (treatment). Differences are within what two seeds can show.
2. **Same flags and seed are not reproducible.** `mb_fresh_s0` (1 bee, same flags, 30k) arrived at 24,524; `mb1_s0` at 29,119. So run-to-run noise alone is about 5k steps for one seed. Thread timing makes runs non-deterministic.
3. The 3-bee control seed 1 that never arrived was the unlucky outlier; 1 bee on seed 1 does arrive (37.8k).
4. Wall clock: 1 bee ran at about 49 steps/s, 3 bees at 21-28 on this Kaggle CPU, so the 3-bee runs were slower per step here.

## Honest status
The central question (do parallel bees plus pheromones beat one agent in training steps) is, so far, answered "not detectably", n=2 seeds per arm and noise of several thousand steps. Not yet tried: more seeds, a longer-lived/stronger success trace, bees at different start states (every episode is the same scenario 0, so extra bees explore the same route).

## Next: stronger pheromone signal (code change 2026-11-04)
- New flags in `multi_bee_training.py` (defaults unchanged): `--novice-success-vigor V` (NOVICE arrivals write success dances at vigor V; implies `--novice-writes`) and `--eta-success X` (weight of the success term, config default 0.5). Both are recorded in `run_config.json`. Stub-tested for plumbing only (the stub never arrives, so the strong write itself was not exercised). Zip: `bee_multi_v8.zip`.
- Why the old signal was weak: shaping was under 1% of the return before the first arrival and 5-7% after, with FORAGER/SCOUT vigors of 0.8-0.95. So raising NOVICE vigor alone cannot give more than that; `--eta-success` is the lever.
- Success traces exist only after the first arrival, so pheromones cannot move the first arrival. The right metric is steps from first arrival to a reliable arrival rate (for example 5 of the last 10 episodes), plus total arrivals, not first-arrival step.
- Proposed arm: `--bees 3 --read-start 5000 --novice-success-vigor 0.9 --eta-success 2.0`, seeds 0 and 1, compared with the existing 3-bee controls. This adds two tuned values; disclose them.
