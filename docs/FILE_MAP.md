# File map: project-bee

Every file in the project folder except `venv/` and `.git/` internals, with what each one contains. Snapshot taken 2026-10-20. Status is one of:

- **tracked**: in git.
- **untracked**: on disk, not in git, not ignored. Commit or ignore it.
- **ignored**: on disk, excluded by `.gitignore`.
- **modified**: tracked, with uncommitted changes.

Paths are relative to `project-bee/` unless shown otherwise.

## Folder tree

```
project-bee/
├── .gitignore
├── Claude.md                    (project instructions for Claude Code)
├── bee_colony_checklist.md      (task checklist)
├── docs/
│   ├── STATUS.md                (where we are: observations, decisions, open items)
│   └── FILE_MAP.md              (this file)
├── Documentation/
│   ├── rough_draft.md           (article draft, personal narrative)
│   └── shaping_spec.md          (shaping spec, frozen settings, pilot results, addendum)
├── sandbox_bee2_training.py     (single-bee SAC training loop)
├── reputation_tracker.py        (E_t, castes, vigor)
├── pheromone_table.py           (spatial trace store)
├── shaping.py                   (potential and shaped-reward functions)
├── measure_beta.py              (beta rule)
├── render_validation.py         (rendered evaluation of a run)
├── idm_pheromone_validation.py  (IDM autopilot validation)
├── test_pheromone_table.py      (9 tests)
├── test_shaping.py              (9 tests)
├── test_sequential_envs.py      (old sanity check: sequential MetaDrive envs)
├── test_parallel_plumbing.py    (old sanity check: multiprocessing workers)
├── test_mixed_policy.py         (old multi-agent check)
├── verify_metadrive.py          (old multi-agent sanity check)
├── run_bee_fleet_v0.py          (obsolete multi-agent approach)
├── checkpoints/                 (ignored: one folder per run)
│   ├── h200_51k_seed0/
│   ├── h500_40k_seed0/
│   ├── penalty20_20k/
│   ├── penalty40/
│   ├── pilot_live_s0/  pilot_live_s1/
│   ├── pilot_snap10_s0/  pilot_snap10_s1/
│   ├── pilot_static_s0/
│   ├── shaping_none_s0/  shaping_none_s1/
│   └── smoke_*/                 (short smoke-test runs, deletable)
├── logs/                        (ignored: training and validation logs, status markers)
├── __pycache__/                 (ignored bytecode; one tracked file, see below)
├── pheromone_table.pkl          (ignored: top-level copy of the h500 40k table)
├── sandbox_bee_checkpoint.pt    (ignored: top-level copy of the h500 40k weights)
├── idm_pheromone_table.pkl      (ignored: old IDM table, pre-fix)
├── idm_validation_table.pkl     (ignored: IDM table from the 8/8 live run)
└── venv/                        (ignored: Python 3.10.11 environment, 44,701 files)
```

## Tracked code

**`sandbox_bee2_training.py`** (tracked, modified in last commits; 29 KB). The single-bee SAC training loop, and the main entry point.
- Holds the SAC actor, twin critic, replay buffer, and the SAC update (`SACAgent`).
- `classify_terminal` and `is_real_terminal`: map MetaDrive's `info` flags to an episode outcome. Only arrive and crash flags count as real terminals.
- `write_episode_traces`: the write hook. Arrival writes a success trace along the path (NOVICE arrivals write none). A crash writes one warning. Vigor ≤ 0 is skipped.
- `save_checkpoint`: writes `sandbox_bee_checkpoint.pt` and `pheromone_table.pkl` into a run folder.
- `run_sac_training`: the loop. Options: `--run-name`, `--steps`, `--seed`, `--warmup`, `--half-life-frac`, `--shaping none|snapshot|live|static`, `--snapshot-every`, `--beta`, `--lam`, `--fixed-vigor` (pilot stub), `--read-start`. Writes `run_config.json` into the run folder. Refuses to reuse a non-empty folder.
- Module constants: `REPUTATION_SMOOTHING = 0.1`, `REPUTATION_HYSTERESIS = 0.05`, `HERE` (script folder, used to anchor paths).

**`reputation_tracker.py`** (tracked, 6 KB). Per-bee reputation.
- `ReputationTracker(agent_ids, smoothing, hysteresis)`.
- `update(agent_id, success)`: E = (1 − smoothing)·E + smoothing·outcome. Arrival = 1; crash, out-of-road, and timeout = 0.
- `caste(agent_id)`: NOVICE below 0.3, FORAGER from 0.3, SCOUT from 0.8. Hysteresis: demote only when E falls 0.05 below the threshold.
- `vigor(agent_id)`: NOVICE 0.05; FORAGER tanh(2E) clipped to [0.1, 0.8]; SCOUT 0.95.
- A `__main__` self-test (passes).

**`pheromone_table.py`** (tracked, 6.5 KB). Spatial trace store.
- `PheromoneTable(grid_size=2.0)`: a dict keyed by 2 m grid cell. Each cell has `success` and `warning` channels, each holding vigor, timestamp (training step), meta, speed, and steer.
- `write(location, trace_type, vigor, step, meta, speed, steer)`: overwrites only if the new vigor is at least the stored one. Rejects vigor ≤ 0.
- `read_nearby(location, step, decay_rate, sensor_radius, sigma)`: max over cells of vigor × Gaussian spatial weight × exponential decay. `decay_rate` has no default on purpose.
- `decay_rate_from_half_life(half_life_steps)`: converts a half-life in steps to a per-step rate.
- `summarize()` and `print_summary()`: plain-English dump.

**`shaping.py`** (tracked, 1.4 KB). Pure functions, no table reads.
- `potential(success, warning, beta, lam)`: Φ = β(S − λW).
- `shaping_term(phi_s, phi_next, gamma, next_is_terminal)`: F = γΦ(s′) − Φ(s), with Φ(s′) = 0 at a real terminal.
- `shaped_reward(env_reward, phi_s, phi_next, gamma, next_is_terminal)`: r′ = r + F.

**`measure_beta.py`** (tracked, 5 KB). Implements the β rule (spec §10a). Loads a checkpoint and a table, rescales every trace to vigor 1.0, runs rollouts, and prints the β such that mean |F| = 10% of mean |r|.

**`render_validation.py`** (tracked, 5.4 KB). Loads `checkpoints/<run-name>/` and runs deterministic episodes with rendering. Prints the distance and sensed traces along the way.

**`idm_pheromone_validation.py`** (tracked, 9.7 KB). Runs MetaDrive's IDM autopilot through the same write hook, with the tracker driving castes. Runs 6 episodes, performs 8 checks, and writes `idm_validation_table.pkl`. Last run 8/8.

**`test_pheromone_table.py`** (tracked, 10.8 KB). 9 tests for the pheromone table: grid keys, overwrite rule, channels, spatial falloff, empty reads, step-based decay, zero-vigor rejection, half-life conversion, and speed and steer.

**`test_shaping.py`** (tracked, 4.6 KB). 9 tests for `shaping.py`: definition, rejection of negative weights, terminal Φ = 0, timeout handling, zero table, r′ = r + F, the return-difference identity, the closed-loop check, and the gamma range.

**`test_sequential_envs.py`** (tracked, 2.7 KB). Old sanity check that several independent `MetaDriveEnv` instances can run one after another. Not part of the current pipeline.

**`test_parallel_plumbing.py`** (tracked, 5.9 KB). Old check for multiprocessing workers, queues, and DONE sentinels on Windows (`spawn`). Not part of the current pipeline. Written for the multi-agent design.

**`test_mixed_policy.py`** (tracked, 4.4 KB). Old multi-agent check. It imports `MultiAgentIntersectionEnv`, from the abandoned approach.

**`verify_metadrive.py`** (tracked, 5 KB). Old sanity check for the multi-agent environment.

**`run_bee_fleet_v0.py`** (tracked, 12.4 KB). Obsolete. Wires the tracker into the multi-agent intersection env. It still constructs the tracker with `step_up` and `step_down`, which no longer exist, so it can't run as written.

**`.gitignore`** (tracked, 55 bytes, UTF-8). Ignores `venv/`, `__pycache__/`, `*.pyc`, `checkpoints/`, `*.pt`, `*.pkl`, `logs/`.

## Tracked documents

**`Claude.md`** (tracked, modified, uncommitted; 14 KB). Project instructions for Claude Code. Covers the idea, what is out of scope, the architecture, the file table, the current state (with a pointer to `docs/STATUS.md`), MetaDrive facts learned the hard way, and the working style. Windows treats `CLAUDE.md` and `Claude.md` as the same file, but git tracks `Claude.md`.

**`bee_colony_checklist.md`** (tracked, 15 KB). Living task list. Sections: foundation, single-bee loop, frozen-policy history, IDM validation, render validation, fixes before comparison runs, decisions to make, reading stage, multi-bee, open items. Top note marks pre-fix runs as exploratory.

**`Documentation/shaping_spec.md`** (tracked, 21 KB). The shaping specification. Sections: setting, potential, reward equation, static vs live, checks, what a trace carries, the single-bee test, cold start (option A, approved), the reputation replay, the EWMA decision, frozen settings (§10), the beta rule (§10a), the seed-0 and seed-1 pilot results (§10b–c), cheap checks (§10d), valley and warnings checks (§10e), and an addendum from the owner (2026-10-19) that is appended without changes above it.

**`Documentation/rough_draft.md`** (tracked, 11 KB). An article draft in the owner's voice. It narrates the path to the project. It is not a source of results.

**`docs/STATUS.md`** (untracked, 9 KB). Status note for resuming work. Observations, decisions, open items, and commands.

**`docs/FILE_MAP.md`** (untracked). This file.

## Tracked artifacts

**`__pycache__/reputation_tracker.cpython-310.pyc`** (tracked, 3.6 KB). Compiled bytecode from the first push. It should not be tracked. Remove it with `git rm --cached`.

## Untracked and ignored files at the top level

**`idm_pheromone_table.pkl`** (ignored, 10 KB, 2026-10-10). Pickled pheromone table from the first IDM validation run, before the fixes. Exploratory. Do not reuse.

**`idm_validation_table.pkl`** (ignored, 12.8 KB, 2026-10-18). Pickled pheromone table from the live IDM validation run (8/8). Its traces carry speed and steer.

**`pheromone_table.pkl`** (ignored, 24 KB, 2026-10-11). Top-level copy of the h500 40k table. Pre-fix. Same as `checkpoints/h500_40k_seed0/pheromone_table.pkl`.

**`sandbox_bee_checkpoint.pt`** (ignored, 2.67 MB, 2026-10-11). Top-level copy of the h500 40k weights. Same as `checkpoints/h500_40k_seed0/sandbox_bee_checkpoint.pt`. The actor, critic, critic target, and log_alpha are stored as a torch dict.

**`__pycache__/*.pyc`** (ignored). Bytecode for `pheromone_table`, `render_validation`, `sandbox_bee1` (an old module name), and `sandbox_bee2_training`. The `sandbox_bee1` file is stale.

## Checkpoints (ignored)

Every run folder holds the same three kinds of file:
- `sandbox_bee_checkpoint.pt`: torch dict with `actor`, `critic`, `critic_target`, and `log_alpha`. Always 2,674,761 bytes, because the architecture is fixed.
- `pheromone_table.pkl`: pickled dict of grid cells, each with `success` and `warning` channels.
- `run_config.json`: the settings the run used. Written from the script; the newer runs include shaping, β, λ, read gate, pilot vigor, and reputation settings.

| Folder | What it is |
|---|---|
| `h200_51k_seed0/` | Backup of the horizon-200 run at about step 50k. Exploratory. No `run_config.json`. |
| `h500_40k_seed0/` | Horizon-500 run at step 40k, pre-fix. The render check used this policy. No `run_config.json`. |
| `penalty20_20k/` | 20k-step run with penalty 20, from an older script version. Exploratory. |
| `penalty40/` | Penalty-40 run that froze. Historical. |
| `pilot_snap10_s0/` | Seed-0 pilot, snapshot arm (K = 10), stub vigor 1.0, β = 2.86. |
| `pilot_live_s0/` | Seed-0 pilot, live arm, same settings. |
| `pilot_snap10_s1/` | Seed-1 pilot, snapshot arm. |
| `pilot_live_s1/` | Seed-1 pilot, live arm. |
| `pilot_static_s0/` | Seed-0 pilot, static arm (table frozen at the gate). Confounded; see spec §10e. |
| `shaping_none_s0/` | Seed-0 reference, no shaping, tracker on. Final checkpoint used for the β rule. |
| `shaping_none_s1/` | Seed-1 reference, no shaping, tracker on. |
| `smoke_*/` | Short smoke runs (2000 steps). Not results. Safe to delete. |

## Logs (ignored)

| File | What it is |
|---|---|
| `logs/train_penalty20.log` | 20k-step run, penalty 20, older script (warmup 1000, no seed, horizon 200). Exploratory. |
| `logs/train_100k_seed0.log` | Horizon-200 run, seed 0. Ends at step 51,127 (stopped; cause not recorded). Exploratory. |
| `logs/train_100k_h500_seed0.log` | Horizon-500 run, seed 0. Ends at step 43,500 with no completion line. Exploratory. |
| `logs/idm_validation.log` | Output of the first IDM validation run (pre-fix). |
| `logs/shaping_none_s0.log`, `logs/shaping_none_s1.log` | Reference arms. |
| `logs/pilot_snap10_s0.log`, `logs/pilot_live_s0.log` | Seed-0 pilot arms. |
| `logs/pilot_snap10_s1.log`, `logs/pilot_live_s1.log` | Seed-1 pilot arms. |
| `logs/pilot_static_s0.log` | Seed-0 static arm. |
| `logs/pilot_status.txt` | Driver status for the seed-0 pilots. Ends with `ALL_DONE`. |
| `logs/seed1_status.txt` | Driver status for the seed-1 batch. Ends with `ALL_DONE`. |
| `logs/static_status.txt` | Driver status for the static run. |

Logs are not written by the code. They come from the `>` redirect in the shell. Use `python -u` so the log is not block-buffered.

## Virtual environment (ignored)

`venv/` is a Python 3.10.11 environment with 44,701 files. It holds MetaDrive `metadrive-simulator` 0.4.3, torch 2.13.0 (CPU), matplotlib 3.10.9, and their dependencies. Some `Scripts/*.exe` files are left from an earlier Python 3.14 install (for example `pip3.14.exe`). Don't edit anything under it. To rebuild it, use Python 3.10, not 3.14.

## Outside the project

- `C:\Users\jeevika\metadrive-sim\backups\pilot-single-bee-v1-2026-10-18\`: copies of `logs/` and `checkpoints/` as of the pilot. Git ignores both folders.
- `C:\Users\jeevika\AppData\Local\Temp\claude\...\scratchpad\`: one-off analysis scripts (`phi_check.py`, `phi_check2.py`, `first15m.py`, `by_steps.py`, `none_variance.py`, `seed1_analysis.py`, `reasons.py`, `pilot_analysis.py`, `measure_beta_*.log`) and `outcomes_h500.txt`. Not part of the repo. The owner plans to rewrite the Φ-map and steps-aligned comparison in the repo.

## Tags and commits

- Tags: `exploratory-v0` = `8200c19` (pre-fix code). `pilot-single-bee-v1` = `39ae790` (pilot code and results).
- HEAD: `4fd0962` (spec addendum).
- Uncommitted: `Claude.md` (pointer to `docs/STATUS.md`), and `docs/` (STATUS.md and FILE_MAP.md).
