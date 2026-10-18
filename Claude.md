# Project: Bee-Colony-Inspired Parallel Exploration for RL Driving (project-bee)

Read this first. Current task status lives in `bee_colony_checklist.md` (source of truth for what is done and what is next). This file explains the idea, the decisions already made, and the traps already hit, so you don't re-litigate or re-discover them.

## The idea in one paragraph

Single-agent SAC training for a driving policy (the owner's separate main project) was slow and sample-inefficient. This project tests whether **many parallel, independent "bee" workers sharing ONE policy network, plus a lightweight shared spatial "pheromone" table, can reach a given driving competence in fewer training steps** than a single agent. It is a **training-time technique only**. At test time there is one bee, alone, with no pheromones and no fleet, and it has to drive the route. Inspiration: honeybee colony division of labor (scouts discover, experienced foragers guide others) and China's vehicle-infrastructure information sharing.

Research question: does experience-weighted, reward-side guidance from other workers measurably cut training steps versus the same SAC with no guidance? A clean negative result is acceptable and should be reported honestly.

## What this is NOT (out of scope, already decided)

- Not real-time multi-vehicle coordination or a fleet product. Bees are NOT co-located on one road; each runs in its own independent env instance. (An earlier `MultiAgentIntersectionEnv` approach was abandoned for this reason. `run_bee_fleet_v0.py`, if present, is a leftover from it.)
- Not a C++ rewrite. MetaDrive is Python-only. Parallelism = Python `multiprocessing` (separate processes), not threads.
- Not reusing the main SAC project's checkpoint. That actor takes a 2048-dim CNN+attention latent plus nav_info; this project uses MetaDrive's flat 259-dim vector obs. Incompatible. A fresh small MLP is the right choice here.
- The quorum-sensing / QMIX cross-inhibition "Step 3" from the original theory docs was **dropped** (it was built for whole-hive relocation and doesn't map to independent-destination vehicles). Don't revive it.

## Architecture (current design)

1. **N independent single-agent `MetaDriveEnv` workers** (multiprocessing), no physical interaction.
2. **One shared SAC policy** (parameter sharing: one set of weights, not N models). Actor and twin critic are small MLPs on the 259-dim obs.
3. **Reputation tracker** (`reputation_tracker.py`): per-worker experience `E_t` is an exponentially weighted arrival rate over TERMINAL outcomes only: `E <- (1 - alpha) E + alpha * outcome` (arrive = 1, crash = 0). Horizon timeouts are not updated. It is never credited per step for surviving. `alpha = 0.1` (`REPUTATION_ALPHA` in the training script). `E_t` maps to castes with thresholds NOVICE < 0.3 <= FORAGER < 0.8 <= SCOUT, so FORAGER means roughly a 30% recent arrival rate. Vigor: NOVICE = 0.05 (approved; low, muted for success writes), FORAGER = tanh(2E) clipped to [0.1, 0.8], SCOUT = 0.95. NOVICE arrivals write no success trace. NOVICE crashes still write a warning.
4. **Pheromone table** (`pheromone_table.py`): spatial dict keyed by 2 m grid cell, `(floor(x/2), floor(y/2))`. Each cell has two **independent channels**, `success` (turning guide) and `warning`, each storing vigor/timestamp/meta, overwritten only if new vigor >= existing. `write()` takes the training `step` as its timestamp and rejects vigor <= 0. `read_nearby()` takes `step` and a required `decay_rate` (per training step), returns max-over-cells intensity per channel with Gaussian spatial falloff and exponential temporal decay. `decay_rate_from_half_life()` converts a half-life in steps. `summarize()`/`print_summary()` give a plain-English dump.
5. **Write triggers** (episode end): arrive_dest writes `success` along the whole path; a crash writes one `warning` at the crash location.
6. **Reading is reward-side only.** Nearby traces will shape the reward (bonus for succeeding near a trace). **Never blend guidance into the action**: the policy would learn to lean on guidance that doesn't exist at solo test time. Reading is gated: writing can run from step 1, reading only activates after a step-count threshold, so the network isn't biased by noise from equally clueless early bees.
7. **Vigor is currently stubbed to 1.0** in the single-bee training loop (a hierarchy is meaningless with one bee). Swap in `tracker.vigor(worker_id)` once multiple bees exist.

Not built yet: the waggle-dance message layer (symbolic `[bearing, distance, vigor]` vectors with encode/decode/vigor-weighted integration) from the original theory docs. The pheromone table currently carries guidance. Treat waggle messaging as a possible later layer, not an assumption.

Planned refinement, not built: scale deposited vigor by local maneuver difficulty, `Vdep = Vagent * (1 + B * |steer|)`.

## Files

| File | Status | Purpose |
|---|---|---|
| `reputation_tracker.py` | done, tested (not wired into training yet) | E_t / caste / vigor |
| `pheromone_table.py`, `test_pheromone_table.py` | done, 8 tests pass | spatial trace store; `write(..., step)`, `read_nearby(..., step, decay_rate)` |
| `test_sequential_envs.py` | done | multiple independent envs, clean lifecycle |
| `test_parallel_plumbing.py` | done | multiprocessing workers + queue + sentinels |
| `sandbox_bee2_training.py` | working, pre-comparison | single-worker SAC loop, real-terminal buffer pushes, pheromone write hook, per-run checkpoints in `checkpoints/<run-name>/` |
| `render_validation.py` | working | loads `checkpoints/<run-name>/`, rendered deterministic episodes |
| `idm_pheromone_validation.py` | working (not re-run since the fixes) | IDM autopilot through the same write hook; writes `idm_validation_table.pkl` |
| `verify_metadrive.py`, `test_mixed_policy.py` | older checks, not read in the latest review | MetaDrive sanity checks |
| `run_bee_fleet_v0.py` | obsolete | old co-located multi-agent approach |
| `Documentation/rough_draft.md` | theory draft, unverified claims | not a source of results |

## Current state (update this section as it changes)

- Baseline SAC loop runs end to end. Fixed so far: alpha/entropy collapse (alpha floored at 0.03 via in-place `log_alpha.data.clamp_` AFTER the alpha optimizer step); reward hacking (MetaDrive defaults of 5.0 crash/out-of-road penalty let fast-then-crash episodes bank 20-30 reward).
- As of commit `4822987`. Tag `exploratory-v0` (`8200c19`) marks the pre-fix code. Runs made before it are exploratory, not comparison baselines.
- Penalty history: 40.0 froze the policy (historical). Current value is **20.0** for out_of_road, crash_vehicle, and crash_object. Horizon **500**. Map C. Traffic density, `num_scenarios` (1), and `start_seed` (0) are MetaDrive defaults. Every episode is Scenario Index 0, so the policy learns one fixed route.
- The h200 (horizon 200) and h500 (horizon 500) 100k-step runs both drive. The h500 run had its first arrival at episode 148. The h500 log stops at step 43500. Its last checkpoint is 40k.
- Fixes since the exploratory runs (all committed): real-terminal buffer pushes (`2b56049`); decay by training steps (`059ab93`); env closed in `finally`, outputs anchored to the script folder, zero-vigor writes rejected (`4822987`). Each run writes to `checkpoints/<run-name>/` and refuses to reuse a folder that has files.
- Pheromone half-life is a fraction of total steps, set with `--half-life-frac` (default 0.15; 0 = no decay). It is recorded in the run config but has no effect until `read_nearby()` is wired into training.
- Render check of the 40k h500 policy (`checkpoints/h500_40k_seed0/`, deterministic): episodes 1–3 each crash at step 157–162 after 195–200 m, on the same route. Episode 4 stopped at step 30 with exit code 0 and no error. The cause is unknown and episode 5 did not run. The visual check is not done.
- Cold start decided (option A, `Documentation/shaping_spec.md` §8): NOVICE warns at vigor 0.05 and never writes success. `E_t` is terminal-only. The tracker is wired into the single-bee loop: each episode writes with the caste the bee had before that episode, then folds the outcome into `E_t`.
- Reputation is an EWMA arrival rate, chosen after the step-based rule was shown not to promote on the h500 outcomes (see spec §8a). Replay of the h500 outcomes at alpha 0.1: first FORAGER at episode 181, 6 episodes in FORAGER, no SCOUT. Note: 0.05, 0.1, and 0.2 were all replayed before 0.1 was chosen, so the choice is not a clean a priori one. Fix alpha in the spec before the comparison runs.
- `run_bee_fleet_v0.py` (obsolete) still constructs the tracker with `step_up`/`step_down`, so it no longer runs. Leave it alone, or delete it.
- Not done: baselines on the fixed code; IDM validation on the fixed code; reading not wired; no multi-worker.
- The top-level `sandbox_bee_checkpoint.pt` and `pheromone_table.pkl` are copies of `checkpoints/h500_40k_seed0/` (SHA-256 matched). Do not write new runs to the top level.
- Commands: `python sandbox_bee2_training.py --run-name NAME --steps N [--warmup W] [--half-life-frac F] [--seed S]`; `python render_validation.py --run-name NAME [--episodes K]`.

## MetaDrive facts learned the hard way (don't rediscover)

- Environment: Python 3.10 venv (3.14 breaks old pinned deps), `metadrive-simulator` 0.4.3, Windows.
- Import: `from metadrive import MetaDriveEnv` (and `MultiAgentIntersectionEnv`). Deep submodule paths don't exist.
- `MetaDriveEnv` obs is a flat `(259,)` array. `reset()` may return `(obs, info)`; `step()` may return 4 or 5 values. Handle both.
- Ego position: `env.agent.position`. `env.vehicle` still works but is deprecated. There are NO `"x"`/`"y"` keys in `info` (using `info.get("x", 0.0)` silently collapses every path to the origin).
- Terminal flags in `info`: `arrive_dest`, `out_of_road`, `crash_vehicle`, `crash_object`, `crash_building`. There is NO generic `"crash"` key (checking it silently never fires). Use `env.agent.speed_km_h` for speed.
- At max horizon MetaDrive sets both terminated and truncated. Only real terminals (`arrive_dest`, `out_of_road`, `crash_*`) are pushed to the replay buffer as `done=1` (`is_real_terminal`). Horizon timeouts still end the episode for reset and traces, but they bootstrap. Fixed in `2b56049`.
- The engine is a process-wide singleton. A failed/unclosed env makes the next one throw "Can not call this API after engine initialization!". Always `env.close()` in `finally`.
- `agent_configs` does not accept arbitrary per-agent policy keys. Mixed per-agent policies via config are not supported; external action dicts are used instead.
- Multiprocessing on Windows uses `spawn`: guard entry with `if __name__ == "__main__":`, import MetaDrive inside the worker function, keep worker targets as top-level functions, drain the result queue while workers run (don't `join()` first), have each worker send a final DONE sentinel (also after errors; the parent retires a worker only on DONE).
- Modifying the alpha clamp by reassigning `log_alpha` orphans it from its optimizer. Clamp `.data` in place.
- Pheromone timestamps are training steps (since `059ab93`). Pickles from before that store wall-clock seconds. Reading them with decay on silently gives ~0. Those pickles are exploratory only.
- `read_nearby()` has no default `decay_rate` on purpose: a silent default is how the old per-second 0.002 leaked into the rate.
- Warning and success channels can overlap. At the 40k h500 crash spot (87, -111) the warning cell and a success cell coincide, because arrival paths drive through it. A warning there does not mean "never drive here".

## Working style (important)

- The owner is an ML beginner learning first-principles and **writes the code himself**. Default to asking questions, explaining the "why," and reviewing his code, not dumping full implementations. Give full code only when he asks for it ("give me the fixed file").
- Build in small standalone pieces with a `__main__` test block, verify each in isolation, then wire together. This discipline has caught several silent bugs; keep it.
- When reviewing, list bugs in the order Python would hit them, and say which are crashes versus silent-wrong-data. Be direct about problems.
- Do not invent citations. The original theory docs contain unverified claims (e.g. a "100-neuron decoder, Edinburgh 2024" reference and a Wikipedia entry cited as a source). Verify or omit. Stigmergy, pheromone trails, and gated/sparse communication are established prior art, not novel. The defensible novel angle is experience-graded (temporal-polyethism-style) guidance weighting instead of a fixed scout/worker caste split, applied as reward-side guidance during parallel training.
- He works in bursts and steps away for days or weeks. When he returns, start from the checklist and ask where he left off rather than assuming.

## Separate project (do not touch from here)

The main single-agent SAC driving project lives in a sibling folder (`metadrive-sim/project`) with its own venv. Don't modify its environment or checkpoints.