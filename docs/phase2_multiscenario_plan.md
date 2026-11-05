# Phase 2 plan: multi-scenario training, 2x2 (bees x pheromones), 2 seeds (2026-11-05)

Question: does 3-bee parallelism and/or per-scenario pheromone guidance help a policy learn to drive across several roads, and does it generalize to roads it never trained on?

Design (full 2x2, Option A):

| | Pheromones ON (`--read-start 5000`) | Pheromones OFF (`--read-start 999999`) |
|---|---|---|
| 1 bee | mb1_phe | mb1_nophe |
| 3 bees | mb3_phe | mb3_nophe |

- Seeds 0 and 1 per arm, 100k steps, `--scenario-seeds 0,1,2,3,4` (each episode draws a seed at random with replacement from the worker's own RNG), `--pheromone-per-scenario` in all four arms (so structure is identical; with reading off it only affects where writes go).
- Other flags as Phase 1: `--chunk-steps 5 --max-lag 10 --sync-every 5 --warmup-steer 0.3 --warmup-throttle-min 0.2 --buffer 200000 --novice-success-vigor 0.15 --eta-success 0.75`. Half-life = 15% of 100k = 15k steps.
- Eval (`eval_policy.py`): frozen `checkpoint_100k` (checkpoint rule fixed before looking: 100k only), deterministic actions, no table, 20 episodes on each of seeds 0, 2 (trained roads) and 5, 6, 7 (held out). Same shuffled schedule for every arm (`--seed 0`). Train and held-out are reported separately.

Implementation (multi_bee_training.py): `--scenario-seeds`, `--pheromone-per-scenario`; env built with `start_seed=min, num_scenarios=max-min+1` and `env.reset(seed=k)` (verified legal only inside that range); learner keeps `{key: PheromoneTableV2}` with key = scenario seed (per-scenario) or 0 (global), broadcasts the whole dict, each worker swaps `reader.table` to its chosen seed's table; checkpoints save the dict when per-scenario, the single table otherwise (old format); `episodes.csv` already has `scenario_seed`; `summary.json` gains `scenarios` (per-seed training stats). Smoke-tested only (2 bees, 3k steps): seeds all drawn, tables keyed per seed, eval script runs.
No `PheromoneManager` class: a dict does the same job.

Known limits: 2 seeds per arm (Phase 1 outcomes were bimodal); routes differ (about 116-287 m) so compare route completion, not raw distance; 100k steps may give no arrival on some arms; wall clock about 5.5 h on Kaggle CPU.
