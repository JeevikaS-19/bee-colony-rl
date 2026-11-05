# Phase 1 training results: 1 bee vs 3 bees, scenario 0, 3 seeds each (2026-11-05)

Sources: `kaggle-checkpoints/phase_1_res/multibee_checkpoints/{mb1,mb3}_train_seed{0,1,2}/` (episodes.csv, summary.json, checkpoints 10k-60k, final/). Code at commit `2d588e2`.
Flags (all six runs): `--steps 60000 --chunk-steps 5 --max-lag 10 --sync-every 5 --warmup-steer 0.3 --warmup-throttle-min 0.2 --buffer 200000 --read-start 5000 --novice-success-vigor 0.15 --eta-success 0.75`, `--bees 1` or `3`, `--seed 0|1|2`. All six finished with no worker errors. Scenario 0 only.

## Per-run results

| Run | First arrival | Arrivals | 5-of-10 reliable at | Arrivals in last 10k | Final castes | steps/s |
|---|---|---|---|---|---|---|
| mb1 seed0 | none | 0 | never | 0 / 25 | NOVICE | 48.2 |
| mb1 seed1 | 46,775 | 42 | 48,098 | 32 / 45 | FORAGER | 44.7 |
| mb1 seed2 | none | 0 | never | 0 / 25 | NOVICE | 45.4 |
| mb3 seed0 | 27,072 | 7 | 29,792 | 0 / 21 | 3 x NOVICE | 37.7 |
| mb3 seed1 | 44,562 | 32 | 47,538 | 24 / 27 | 3 x FORAGER | 35.7 |
| mb3 seed2 | 47,899 | 33 | 49,710 | 28 / 31 | 3 x FORAGER | 34.6 |

"Reliable" = first episode that ends a run of 10 episodes with at least 5 arrivals (episode order, all bees).

Arrival rate per 10k-step window (0-10k ... 50-60k):
- mb1 s0: 0, 0, 0, 0, 0, 0 (mean path 37, 11, 18, 92, 138, 143 m)
- mb1 s1: 0, 0, 0, 0, 0.15, 0.71
- mb1 s2: 0, 0, 0, 0, 0, 0 (max path 147 m, never got near the 230 m route end)
- mb3 s0: 0, 0, 0.18, 0.08, 0, 0 (mean path 184 m in the last window, i.e. nearly to the end, but no arrivals)
- mb3 s1: 0, 0, 0, 0, 0.30, 0.89
- mb3 s2: 0, 0, 0, 0, 0.19, 0.90

## Observations

1. **Outcomes are bimodal, not a smooth shift.** A run either takes off around 44-50k steps and reaches 70-90% arrival in the last window (mb1 s1, mb3 s1, mb3 s2), or it does not become reliable (mb1 s0, mb1 s2, mb3 s0).
2. **3 bees reliable in 2 of 3 seeds, 1 bee in 1 of 3.** With n=3 per arm this is not a detectable difference. Last-window arrival rate in the successful runs is 0.89 / 0.90 (3 bees) versus 0.71 (1 bee), also one run against two.
3. **mb3 seed0 repeats the collapse seen in `mb3_strong_s0`.** It arrived early (27.1k, the earliest of any run here), had a burst of 7 arrivals between 20k and 40k, then 0 after that. In the 50-60k window the mean path was 184 m, close to the ~230 m end, but never arrived. All three bees stayed NOVICE (E never reached 0.3).
4. **1-bee seeds 0 and 2 never arrived** in 60k steps, which is a worse failure than run 9's 1-bee (which arrived at 29.1k / 37.8k, but with `--read-start 999999`). Seed 2 got only to 147 m. These runs have pheromone reading on, which run 9's 1-bee did not, so the two are not the same arm.
5. **First arrival is later here than in run 9** for the four runs that arrived at 44-48k (run 9: 27-39k). This is a different config (reading on, eta 0.75, novice success vigor 0.15) and different seeds, so it is not a clean comparison. It does not look like the medium pheromone setting speeds up first arrival, which was not expected to move it anyway (success traces only exist after a first arrival).
6. **The reputation hierarchy activated for the first time in a multi-bee run:** in the three runs that became reliable, the final castes are FORAGER (all 3 bees in each 3-bee run, the single bee in mb1 s1). No SCOUT.
7. **Crash type:** mb1 s1 had 11 `crash_vehicle` terminals (traffic), none in the 3-bee runs. Everything else is out_of_road or timeout.
8. **Speed:** 3-bee runs were 35-38 steps/s versus 45-48 for 1 bee on this Kaggle CPU, so 3 bees is slower in wall-clock per training step (as in run 9).
9. **Success writes:** equal the arrival count in every run (42, 7, 32, 33), so the NOVICE success write at vigor 0.15 fired on every arrival. Pheromone table size at the end: 67 and 86 cells in the two 1-bee runs with no arrival (crash warnings only), 322-364 cells in the others.

## Caveats
- n=3 per arm, one scenario, run-to-run noise of several thousand steps. The right reading is "no detectable difference in whether the policy becomes reliable", with a lean toward 3 bees that two successes versus one cannot support.
- No no-pheromone control in this batch (`--read-start 999999`), so nothing here isolates pheromones from the bee count.
- Checkpoints land slightly after 40k/50k/60k (the exact step is stored in agent.pt).
- All numbers are computed from episodes.csv and summary.json; nothing here is from the render check.

## What this means for Phase 2 (eval)
- Only mb1 s1, mb3 s1 and mb3 s2 have a competent final policy on scenario 0, so only those can say anything about generalization to unseen roads. The other three would be near 0% on seed 0 as well, which is a floor, not a generalization result.
- The 40k checkpoints come before any run took off (first reliable arrival is 48k at the earliest apart from mb3 s0), so 40k is a pre-competence snapshot; 50k and 60k are the useful ones. mb3 s0 is a special case: its 30k-40k checkpoint was competent, its 60k one is not.
- Decide the checkpoint rule before looking at eval results.
