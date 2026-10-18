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
3. **Reputation tracker** (`reputation_tracker.py`): per-worker experience `E_t`, which grows slowly and ONLY on terminal outcomes (arrive = success, crash = failure). It is never credited per step for surviving. `E_t` maps to castes with deliberately discontinuous thresholds (NOVICE < 0.3 <= FORAGER < 0.8 <= SCOUT). Vigor: NOVICE = 0 (muted, listens only), FORAGER = tanh(2E) clipped to [0.1, 0.8], SCOUT = 0.95. Discontinuity is intentional (better noise blocking).
4. **Pheromone table** (`pheromone_table.py`): spatial dict keyed by 2 m grid cell, `(floor(x/2), floor(y/2))`. Each cell has two **independent channels**, `success` (turning guide) and `warning`, each storing vigor/timestamp/meta, overwritten only if new vigor >= existing. `read_nearby()` returns max-over-cells intensity per channel with Gaussian spatial falloff and exponential temporal decay. `summarize()`/`print_summary()` give a plain-English dump.
5. **Write triggers** (episode end): arrive_dest writes `success` along the whole path; a crash writes one `warning` at the crash location.
6. **Reading is reward-side only.** Nearby traces will shape the reward (bonus for succeeding near a trace). **Never blend guidance into the action**: the policy would learn to lean on guidance that doesn't exist at solo test time. Reading is gated: writing can run from step 1, reading only activates after a step-count threshold, so the network isn't biased by noise from equally clueless early bees.
7. **Vigor is currently stubbed to 1.0** in the single-bee training loop (a hierarchy is meaningless with one bee). Swap in `tracker.vigor(worker_id)` once multiple bees exist.

Not built yet: the waggle-dance message layer (symbolic `[bearing, distance, vigor]` vectors with encode/decode/vigor-weighted integration) from the original theory docs. The pheromone table currently carries guidance. Treat waggle messaging as a possible later layer, not an assumption.

Planned refinement, not built: scale deposited vigor by local maneuver difficulty, `Vdep = Vagent * (1 + B * |steer|)`.

## Files

| File | Status | Purpose |
|---|---|---|
| `reputation_tracker.py` | done, tested | E_t / caste / vigor |
| `pheromone_table.py`, `test_pheromone_table.py` | done, 5 tests pass | spatial trace store |
| `test_sequential_envs.py` | done | multiple independent envs, clean lifecycle |
| `test_parallel_plumbing.py` | done | multiprocessing workers + queue + sentinels |
| `sandbox_bee2_training.py` | working, being tuned | single-worker SAC loop + pheromone write hook + checkpoint/pickle save |
| `render_validation.py` | working | loads checkpoint + table, runs rendered deterministic episodes |
| `run_bee_fleet_v0.py` | obsolete | old co-located multi-agent approach |

## Current state (update this section as it changes)

- Baseline SAC loop runs end to end. Fixed so far: alpha/entropy collapse (alpha floored at 0.03 via in-place `log_alpha.data.clamp_` AFTER the alpha optimizer step); reward hacking (MetaDrive defaults of 5.0 crash/out-of-road penalty let fast-then-crash episodes bank 20-30 reward).
- **Open problem:** raising out_of_road/crash_vehicle/crash_object penalties to 40.0 overcorrected. Rendered eval shows a frozen policy (about 0.3 km/h, constant braking, ~0.1 m travelled in 130 steps). Standing still is now the safest strategy. Next steps: retune penalty to a middle value (try 15-20), log distance travelled per episode in training, and if it still freezes add a small idle/no-progress penalty. Change one variable at a time.
- Then: re-run render validation on a policy that actually drives, then enable reading, then scale to multiple workers sharing the network.

## MetaDrive facts learned the hard way (don't rediscover)

- Environment: Python 3.10 venv (3.14 breaks old pinned deps), `metadrive-simulator` 0.4.3, Windows.
- Import: `from metadrive import MetaDriveEnv` (and `MultiAgentIntersectionEnv`). Deep submodule paths don't exist.
- `MetaDriveEnv` obs is a flat `(259,)` array. `reset()` may return `(obs, info)`; `step()` may return 4 or 5 values. Handle both.
- Ego position: `env.agent.position`. `env.vehicle` still works but is deprecated. There are NO `"x"`/`"y"` keys in `info` (using `info.get("x", 0.0)` silently collapses every path to the origin).
- Terminal flags in `info`: `arrive_dest`, `out_of_road`, `crash_vehicle`, `crash_object`, `crash_building`. There is NO generic `"crash"` key (checking it silently never fires). Use `env.agent.speed_km_h` for speed.
- At max horizon MetaDrive sets both terminated and truncated. The training loop currently treats `terminated or truncated` as a terminal `done` when pushing to the replay buffer, so timeouts cut bootstrapping. Known simplification, worth revisiting.
- The engine is a process-wide singleton. A failed/unclosed env makes the next one throw "Can not call this API after engine initialization!". Always `env.close()` in `finally`.
- `agent_configs` does not accept arbitrary per-agent policy keys. Mixed per-agent policies via config are not supported; external action dicts are used instead.
- Multiprocessing on Windows uses `spawn`: guard entry with `if __name__ == "__main__":`, import MetaDrive inside the worker function, keep worker targets as top-level functions, drain the result queue while workers run (don't `join()` first), have each worker send a final DONE sentinel (also after errors; the parent retires a worker only on DONE).
- Modifying the alpha clamp by reassigning `log_alpha` orphans it from its optimizer. Clamp `.data` in place.

## Working style (important)

- The owner is an ML beginner learning first-principles and **writes the code himself**. Default to asking questions, explaining the "why," and reviewing his code, not dumping full implementations. Give full code only when he asks for it ("give me the fixed file").
- Build in small standalone pieces with a `__main__` test block, verify each in isolation, then wire together. This discipline has caught several silent bugs; keep it.
- When reviewing, list bugs in the order Python would hit them, and say which are crashes versus silent-wrong-data. Be direct about problems.
- Do not invent citations. The original theory docs contain unverified claims (e.g. a "100-neuron decoder, Edinburgh 2024" reference and a Wikipedia entry cited as a source). Verify or omit. Stigmergy, pheromone trails, and gated/sparse communication are established prior art, not novel. The defensible novel angle is experience-graded (temporal-polyethism-style) guidance weighting instead of a fixed scout/worker caste split, applied as reward-side guidance during parallel training.
- He works in bursts and steps away for days or weeks. When he returns, start from the checklist and ask where he left off rather than assuming.

## Separate project (do not touch from here)

The main single-agent SAC driving project lives in a sibling folder (`metadrive-sim/project`) with its own venv. Don't modify its environment or checkpoints.