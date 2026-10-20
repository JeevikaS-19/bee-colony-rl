# STATUS: where the project is (as of 2026-10-20)

Read this first when resuming. It lists the observations so far, what is decided, what is open, and how to run things. The detailed rules are in `Documentation/shaping_spec.md` (frozen settings in §10, addendum at the end). The task list is `bee_colony_checklist.md`. Design background is in `Claude.md`.

## Repo

- Path: `C:\Users\jeevika\metadrive-sim\project-bee`, branch `main`.
- HEAD: `4fd0962` (spec addendum). Working tree clean when this was written, apart from this file.
- Tags: `exploratory-v0` = `8200c19` (pre-fix code, runs before it are not baselines). `pilot-single-bee-v1` = `39ae790` (the pilot code and results).
- Nothing is pushed. Hold the push until `Documentation/rough_draft.md` has been read.
- Backups of `logs/` and `checkpoints/` as of the pilot: `C:\Users\jeevika\metadrive-sim\backups\pilot-single-bee-v1-2026-10-18\`. Git ignores both folders.
- The venv is `project-bee\venv` (Python 3.10.11, MetaDrive 0.4.3). Use `./venv/Scripts/python.exe -B` from Git Bash.
- `Documentation/` holds theory docs and the shaping spec. This folder (`docs/`) holds status notes only.

## Code state

| File | What it does |
|---|---|
| `sandbox_bee2_training.py` | Single-bee SAC loop. Options: `--run-name` (refuses non-empty folders), `--shaping none|snapshot|live|static`, `--snapshot-every`, `--beta`, `--lam`, `--fixed-vigor` (pilot stub, tracker off), `--read-start` (default = end of warmup), `--half-life-frac` (default 0.15). Writes `checkpoints/<run>/` and `run_config.json`. |
| `reputation_tracker.py` | E_t = EWMA (smoothing 0.1) of episode outcomes. Arrival = 1, crash, out-of-road, and timeout = 0. Castes NOVICE < 0.3 ≤ FORAGER < 0.8 ≤ SCOUT, with hysteresis 0.05. Vigor: NOVICE 0.05, FORAGER tanh(2E) clipped to [0.1, 0.8], SCOUT 0.95. |
| `pheromone_table.py` | 2 m grid, success and warning channels. `write(loc, type, vigor, step, meta, speed, steer)` rejects vigor ≤ 0. `read_nearby(loc, step, decay_rate, ...)`; decay is per training step. |
| `shaping.py` | Pure functions: `potential`, `shaping_term`, `shaped_reward`. Φ = β(S − λW), Φ = 0 at real terminals, F = γΦ(s′) − Φ(s). |
| `measure_beta.py` | Implements the β rule (spec §10a): one rollout of a checkpoint with the table rescaled to vigor 1.0. |
| `idm_pheromone_validation.py` | IDM autopilot through the tracker over 6 episodes. Writes `idm_validation_table.pkl`. Last run 8/8. |
| `render_validation.py` | `--run-name` loads `checkpoints/<run>/`. Deterministic rendered episodes. |
| `test_pheromone_table.py` | 9 tests, all pass. |
| `test_shaping.py` | 9 tests, all pass. |
| `run_bee_fleet_v0.py` | Obsolete. Still passes `step_up`/`step_down`, so it no longer runs. Leave it or delete it. |

Top-level `sandbox_bee_checkpoint.pt` and `pheromone_table.pkl` are copies of `checkpoints/h500_40k_seed0/`. Don't write new runs there. `checkpoints/smoke_*` are smoke-test folders and can be deleted.

## Frozen settings (spec §10)

- Reputation: smoothing 0.1, hysteresis 0.05, thresholds 0.3 and 0.8, NOVICE vigor 0.05, NOVICE writes no success traces (option A).
- Shaping: β = 2.86 (from the rule on the `none_s0` final checkpoint, 10 rollouts, 345 cells), λ = 1.0, γ = 0.99. The 10% target is a judgment call.
- Environment: map C, horizon 500, penalties 20.0, MetaDrive defaults for traffic and scenarios (one fixed scenario, `Scenario Index 0`).
- Training: warmup 5000 (throttle-biased), batch 256, lr 3e-4, target entropy 0.0, 60k steps for the comparison runs.

## Observations (what the runs show)

**Reference runs (no shaping, tracker on), seeds 0 and 1:**
- First arrival: `none_s0` at step 27,705; `none_s1` at step 32,553.
- Gap between seeds, 10k windows: 3.4 m at 10k, 1.3 m at 20k, 54.5 m at 30k. The 30k gap is seed-to-seed spread, not shaping.
- Baseline dip at episodes 81–113 appears in both reference runs.

**Stub pilots (fixed vigor 1.0, β = 2.86, shaping gate at step 5000):**
- Plumbing passes. Episodes before the gate match the reference exactly.
- Scale passes. Mean |F| / mean |r| per step is 0.11–0.13.
- Freeze check fails in 3 of 4 shaped runs. Seed 0 snapshot: out-of-road exits near spawn in episodes 161–200. Seed 0 live: the same, in 121–160. Seed 1 live: a timeout stall through 60k steps. Seed 1 snapshot does not stall.
- Steps-aligned, shaped arms lag the reference at 30–40k steps on seed 0 and catch up by 50k. At 30k the seed-0 shaped arms (16 m, 38 m) sit close to `none_s1` (14 m). The seed-0 lag is probably the seed-0 reference running early. Unsupported until there are more seeds.
- `pilot_static_s0` (table frozen at step 5000): finished, 251 episodes, terminal counts out_of_road 181, max_step 65, arrive_dest 5. Not analysed in steps-aligned form. Its table holds only early warnings, so it is confounded.

**Hypotheses checked and not supported:**
- Warnings near spawn are warmup junk. Within 15 m of spawn: 71 policy-phase warnings, 20 warmup. Most are the bee's own crashes.
- A valley at the start. Summed F over the first 15 m of six successful paths: mean −0.06, mixed signs. Against about 30 raw reward over that stretch, it is negligible. (The owner notes the path sum telescopes, so use min(Φ(s_t) − Φ(s_0)); depth about 1.4.)
- The shaping dominates near spawn. Within 10 m of spawn there are 10% of steps and 1% of |F| mass.

**Terminal effect (parked):** arrival steps carry F = −2.85, because Φ(s_T) = 0 at a real terminal while the arrival cell holds a success trace with Φ ≈ +2.85. Seven arrival steps are 0.2% of steps but 9% of |F| mass. Fix later by making Φ = 0 at the goal.

**Other facts:**
- Φ map: spawn Φ = 0.16. Success ridge at and west of spawn (up to about 1.1). Warnings east and south (down to about −1.2). The table has no road geometry, so "leaving the road uphill" is untested. None of 8 rollouts left the road.
- The 40k h500 policy rendered 3 of 5 episodes (crash at about step 160, about 195 m, same route). Episode 4 stopped at step 30, exit 0, cause unknown. Episode 5 not run. Visual check not done.
- Old runs: `train_100k_seed0` (horizon 200) and `train_100k_h500_seed0` drive. Pre-fix, so exploratory only.

## Owner's plan (addendum, awaiting decision)

1. An elder bee trains alone until average route completion ≥ 30%.
2. Then 3–5 learners are introduced and eventually become teachers.
3. A small event bonus when a bee crosses a warned spot without crashing. Not potential-based, and it still needs a precise definition.
- Proposed (not decided): reputation as an EMA of per-episode route completion (arrival = 1.0, FORAGER 0.3 matches 30%). Passed-safely success traces for progress made. The cohort shares one fresh brain, and the elder keeps her own.
- First experiment (proposed): freeze the elder's table, then one fresh bee with vs without it across seeds. Count the elder's steps separately.

## Open items

- Novice success writes (option A says no; the reviewer proposed faint writes at 0.05). Decide for multi-worker.
- Terminal convention at the goal (above).
- Hacking baseline: `none` arms have no shaping log, so the distinct-cell fraction has no comparison.
- Multi-seed: every arm so far is one or two seeds. Reference-to-reference spread is large (54.5 m at 30k).
- Render check of the 40k policy: partial.
- Multi-worker build, and the uniform-weight arm.
- `Documentation/rough_draft.md` is unread. Push is on hold until it is read.

## Commands

```bash
cd /c/Users/jeevika/metadrive-sim/project-bee
./venv/Scripts/python.exe -B reputation_tracker.py        # tracker self-test
./venv/Scripts/python.exe -B test_pheromone_table.py      # 9 tests
./venv/Scripts/python.exe -B test_shaping.py              # 9 tests
./venv/Scripts/python.exe -B idm_pheromone_validation.py  # 8 checks, writes idm_validation_table.pkl
./venv/Scripts/python.exe -u -B sandbox_bee2_training.py --run-name NAME --steps 60000 --seed 0 --shaping none
./venv/Scripts/python.exe -B render_validation.py --run-name NAME --episodes 5
./venv/Scripts/python.exe -B measure_beta.py --checkpoint PATH --table PATH --episodes 10
```

Use `python -u` for background runs. Without it, the log is block-buffered and looks stalled. Free memory was about 1.3–1.5 GB of 15.7 GB during the runs. Background waiters get killed under memory pressure, but the training processes survived.

## Analysis scripts

These live in the scratchpad, not the repo: `C:\Users\jeevika\AppData\Local\Temp\claude\C--Users-jeevika-metadrive-sim-project-bee\90eeed02-1f27-48bc-b8bd-16646579cf6b\scratchpad\`. Files: `phi_check.py`, `phi_check2.py`, `first15m.py`, `by_steps.py`, `none_variance.py`, `seed1_analysis.py`, `reasons.py`, `pilot_analysis.py`, and `outcomes_h500.txt`. The owner plans to rewrite the Φ map and the steps-aligned comparison in the repo.
