# Bee-Colony Project Checklist

Living checklist — check things off as you go. Keep this updated across sessions since this project gets picked up and dropped for weeks at a time.

> **Exploratory only:** runs logged before tag `exploratory-v0` (commit `8200c19`) are NOT comparison baselines. They time out as terminal (bootstrapping cut at horizon), decay pheromones by wall-clock time, and came partly from uncommitted code. Re-run baselines after the fixes below.

## Foundation (all done, verified)
- [x] `reputation_tracker.py` — standalone E_t / caste / vigor tracker
- [x] `test_sequential_envs.py` — multiple independent single-agent MetaDrive instances, clean lifecycle
- [x] `test_parallel_plumbing.py` — real multiprocessing, N workers, genuine interleaving, clean teardown
- [x] `pheromone_table.py` + `test_pheromone_table.py` — grid bucketing, overwrite-if-higher-vigor, two independent channels (success/warning), spatial Gaussian falloff, step-based decay, zero-vigor rejection — all 8 test cases passing

## Single-bee training loop
- [x] Fresh SAC actor producing valid actions in one worker (inference only — this was the "sandbox bee" that turned out to have no learning loop)
- [x] Real SAC training loop built: twin critic, replay buffer, target network, alpha auto-tuning
- [x] **Bug fixed:** alpha/entropy collapse (floored at 0.03, confirmed holding flat across a full run)
- [x] **Bug fixed:** reward hacking (crash/out-of-road penalties raised to 40.0, episode rewards now honest — ~2-6 range, no more 20-30 spikes on crashes) — **but this overcorrected, see "Frozen policy" below**
- [x] Pheromone table write-hook wired into episode-end logic (vigor stubbed at 1.0 — single bee, no hierarchy needed yet)
- [x] Position source fix applied: `env.agent.position` (confirmed in `sandbox_bee2_training.py`)
- [x] Checkpoint saving: `torch.save()` of actor/critic/critic_target/log_alpha → `sandbox_bee_checkpoint.pt`
- [x] Pheromone table persistence: pickle of `pheromone_table.table` → `pheromone_table.pkl`
- [x] Re-ran training (20k steps) with both saves in place — files saved Oct 3
- [x] Training script renamed `sandbox_bee1.py` → `sandbox_bee2_training.py` (v2 was built inside the v1 file)

## Frozen policy (current blocker)
- Rendered eval of the 40.0-penalty policy: ~0.3 km/h, constant braking, ~0.1 m travelled in 130 steps — standing still is now the safest strategy
- Saved pheromone table confirms it: only 2 cells, both `warning` (crashes around (6–10, 6–8), episode steps ~90), **zero `success` traces** — the bee never reached the destination in 20k steps
- [x] Retune penalty to a middle value — set to 20.0 (training + render_validation), **full run not done yet**
- [x] Log distance travelled per episode in training (`distance` + `avg_dist_last_20` on each episode line)
- [x] Full 20k-step run at 20.0 (Oct 10, `logs/train_penalty20.log`): 103 episodes, 94 timeout / 9 crash / 0 arrive. Not frozen, but creeping — avg distance 2.6 → peak 5.4 m (~step 11k) → back to 3.4 m, same as an untrained random policy (3-4 m). Alpha hit the 0.03 floor by ~step 13k. Penalty-40 run backed up in `checkpoints/penalty40/`
- Probe (full throttle, no steer): nothing blocks the car; reaches 54 m / 60 km/h by step 60, goes off-road at the curve, still nets **+33** reward even after the −20 penalty. Driving clearly pays — this is an exploration/learning problem, not the env. The earlier "reward hacking" (20-30 reward on fast-then-crash episodes at penalty 5) was likely real progress reward, not hacking
- [x] Alpha flooring: `target_entropy` −2.0 → 0.0 (alpha only falls while entropy > target); entropy now logged next to alpha every 500 steps. Floor (0.03) kept as a safety net
- [x] Throttle-biased warmup: first 5000 steps use random actions, steer U(−0.3, 0.3), throttle U(0.2, 1.0); SAC updates start after warmup
- [x] Episode log now shows which terminal flag fired + max speed; run-end flag counts
- [x] Seeds set (`--seed`, random/numpy/torch); periodic checkpoint every 10k steps
- [x] **Speed fix:** `torch.set_num_threads(2)` — torch's default 10 threads made each update ~300 ms on this CPU (2 threads: ~22 ms)
- [x] 20k penalty-20 checkpoint backed up to `checkpoints/penalty20_20k/`
- [x] 100k run, seed 0 (Oct 10, `logs/train_100k_seed0.log`) — **killed at step ~51k by Claude Code (system low on memory), not a training failure.** Last checkpoint = step 50k in `sandbox_bee_checkpoint.pt`. **The bee drives:** avg distance 3 m → ~90-97 m per 200-step episode by 41-51k, reward ~90-100, max speed ~21-27 km/h, ending on max_step (timeouts) rather than out_of_road — it gets through the curve that a straight drive fails at 54 m. Alpha dipped to 0.032 at ~20k then recovered to 0.16 by 51k (entropy settled just under target 0.0) — no longer stuck on the floor
- [x] Replay buffer memory checked (Oct 11) — **not the RAM problem, no change made.** Obs are already float32 (MetaDrive's `state_obs.py` returns `.astype(np.float32)`), and `next_obs` of one transition is the same array object as `obs` of the next (loop does `obs = next_obs`), so obs aren't stored twice. ~1.4 KB per transition → ~140 MB at 100k transitions, ~280 MB at the 200k cap. The Oct 10 run died because the whole machine was near full (0.7 GB free of 15.7 GB with other projects open), not because of the buffer. Don't convert dtype inside `push()` — two `astype` calls per transition would break the sharing and double it
- [ ] Before the next long run: free RAM (close other projects / Chrome / Creative Cloud)
- [ ] Decide: resume to 100k (needs resume-from-checkpoint support; replay buffer isn't saved) or restart the full run, ideally with horizon 400-500 so arrival is possible
- [x] Horizon raised 200 → 500 (Oct 11) in training + `render_validation.py` (its episode loop now follows the config horizon). 51k h200 checkpoint backed up to `checkpoints/h200_51k_seed0/`
- [ ] Long run at horizon 500 (100k, seed 0) — started Oct 11 with 4.5 GB free, `logs/train_100k_h500_seed0.log`. Check: first arrive_dest? distance / max speed trend, alpha vs entropy
- [x] ~~**Horizon may make arrival impossible:**~~ IDM needs 293 steps to reach the destination on map C, training horizon is 200. Arriving in 200 needs ~45 km/h average from standstill incl. the curve. Consider horizon 400-500 for the next run (not changed yet — one variable at a time)
- [ ] If the 100k run still doesn't drive: seed the replay buffer with IDM demonstrations (as in the main project)

## IDM pheromone validation
- [x] `idm_pheromone_validation.py` — IDM autopilot drives map C through the same `classify_terminal()` + `write_episode_traces()` as training. Oct 10: **6/6 checks pass** — 3/3 arrivals in 293 steps, success traces on exactly the 132 driven cells, on-path reading ≥ 0.97, nothing sensed off-route, no warnings, table doesn't grow on repeat routes. Output in `idm_pheromone_table.pkl`
- [ ] Optional: tighten the off-route check to a point ~15-20 m from the road (current test point is 143 m away, an easy pass)
- [ ] Define the baseline competence target (e.g. avg_dist_last_20 > ~50 m or first arrive_dest) and record the step count it's reached at — that's the number the pheromone version has to beat
- [ ] If it still freezes, add a small idle/no-progress penalty
- Change one variable at a time

## Render-validation stage
- [x] Write a small separate script: load the saved checkpoint + pheromone table (`render_validation.py`)
- [x] Run a handful of episodes with `use_render=True` and `deterministic=True` — done on the frozen policy, which is how the freeze was found
- [~] Re-run on a policy that actually drives: `checkpoints/h500_40k_seed0/` (pre-fix code), `python render_validation.py --run-name h500_40k_seed0`. Episodes 1–3 each crash at step 157–162 after 195–200 m, on the same route. Episode 4 stopped at step 30 with exit 0 and no error (cause unknown). Episode 5 did not run. Re-run this on the fixed code.
- [ ] Visually confirm: do warning traces actually sit at crash-prone spots? Do success traces sit on sensible turning guides? Numeric check: the nearest warning cell is 1.04 m from the crash point, but a success cell sits on the same spot, because arrival paths drive through it. A warning does not always mean "never drive here".

## Reading-enabled stage (after render validation passes)
- [ ] Decide/implement the step-count gate for when `read_nearby()` starts influencing reward (writing can run from step 1; reading waits)
- [ ] Wire `read_nearby()` output into reward shaping (bonus for succeeding near a trace — reward-side only, never action-blending, per earlier decision)
- [ ] Re-run training with reading enabled, compare against the writing-only baseline

## Scaling to multiple bees
- [ ] Give workers real differences (choose one): per-worker exploration temperature (scout/worker idea), different spawn points along the route, or different scenarios. Spawning some workers near the hard turn (~195–200 m along the route) exposes them to the failure zone more often
- [ ] Add a uniform-weight guidance arm (every trace weighted equally, no reputation). Without it, the comparison cannot show that reputation carries any information
- [ ] Wire the shared SAC actor/critic across multiple multiprocessing workers (not yet done — single worker only so far)
- [ ] Swap the stubbed `vigor=1.0` for real `tracker.vigor(worker_id)` once there's more than one bee to have a hierarchy between
- [ ] Confirm experienced bees' traces actually carry more weight/influence than novice bees' traces

## Fixes before comparison runs (done, committed)
- [x] `.gitignore` re-saved as UTF-8, now ignores `*.pt`, `*.pkl`, `checkpoints/`, `logs/` (`8200c19`)
- [x] Tag `exploratory-v0` = `8200c19`; runs before it marked exploratory only (`dc49fa1`)
- [x] Replay buffer stores only real terminals (arrive / out_of_road / crash_*); horizon timeouts bootstrap (`2b56049`). Verified: 7 terminal transitions for 7 crashes, 0 for 3 timeouts
- [x] Pheromone decay counts training steps, not wall-clock. `write(..., step)`, `read_nearby(..., step, decay_rate)`; `decay_rate` has no default (`059ab93`)
- [x] Robustness: env in `try/finally`, outputs anchored to the script folder, zero-vigor writes rejected and skipped in `write_episode_traces` (`4822987`)
- [x] Each run writes to `checkpoints/<run-name>/` and refuses a non-empty folder. Config saved as `run_config.json`
- [x] Half-life is a fraction of total steps (`--half-life-frac`, default 0.15, 0 = no decay). Recorded only until reading is wired
- [x] IDM validation output renamed to `idm_validation_table.pkl` so re-runs can't overwrite the old pickle
- [x] Smoke run verified twice: exit 0, same terminal count and table each time, no accumulation across runs
- [ ] Re-run IDM validation on the fixed code (writes `idm_validation_table.pkl`; currently import-checked only)

## Decisions to make before claiming results
- [x] **Cold start (option A, approved):** NOVICE warns at vigor 0.05, never writes success. `E_t` terminal-only (spec §8)
- [x] **Reputation rule:** EWMA arrival rate, alpha 0.1, wired into the single-bee loop. The step-based rule never promoted on h500 outcomes (spec §8a)
- [x] Smoothing frozen at 0.1 in spec §10 (renamed from alpha). Disclose that 0.05/0.1/0.2 were all viewed on the h500 outcomes before 0.1 was chosen
- [x] Timeouts count as failures to arrive (`sandbox_bee2_training.py`, `tracker.update` on every episode end)
- [x] Hysteresis 0.05 added: replay flips 4 → 2 at smoothing 0.1
- [x] Live IDM validation through the tracker: promotes at episode 5, 8/8 checks (`idm_validation_table.pkl`)
- [x] Shaping function as pure functions with tests (`shaping.py`, `test_shaping.py`, 9 tests)
- [~] Single-bee arms (60k steps, seed 0): `shaping_none_s0` still running as the reference. `shaping_snap10_s0` and `shaping_live_s0` NOT started. Stopped on the scale check: shaping is about 700x smaller than raw reward per step while no FORAGER bee exists (spec §10 pilot finding). Needs a decision on β / novice floor / novice success writes before the shaped arms run
- [x] Pilot (stub vigor 1.0, beta 2.86, seed 0) finished: plumbing passes (identical until step 5000), scale ratio 0.11-0.13 passes
- [ ] Pilot FAILS the freeze check in both shaped arms: early out-of-road exits within ~4 m of spawn after the gate (spec §10b). Mechanism not identified. Next: second seed, read the collapsed episodes' actions, control with the terminal convention changed
- [x] Second seed (seed 1) pilots run: snapshot shows no sustained freeze; live shows a timeout stall (spec §10c). `none_s1` reference still running at 113 episodes when recorded
- [ ] Hacking check needs a baseline: the none arm has no shaping log
- [ ] Decide: may NOVICE bees write faint success traces at 0.05? Reviewer proposes yes; current setting is no (option A). Changing it means re-running the arms
- [ ] Multi-seed: each arm is one seed (seed 0). Add seeds before any claim
- [ ] Confirm the pheromone half-life (0.15 × steps is the first try; ablate with `--half-life-frac 0`)
- [ ] Multi-worker: the decay step counter must be one shared global env-step count across workers, not per-process. Currently single-worker only
- [ ] Hyperparameters for the paper (warmup, throttle bias, `target_entropy=0.0`, horizon, penalties, decay, batch size) must be identical across all arms and listed in the paper

## Known open items, not blocking right now
- [x] (fixed `2b56049`) Timeouts were pushed to the replay buffer as terminal `done`, which cut bootstrapping at the horizon
- [ ] Alpha optimizer is still using the same learning rate as actor/critic (the originally-planned `lr/10` separate rate was never actually applied — the floor alone fixed the visible symptom, but worth knowing if alpha needs further tuning later)
- [ ] Future refinement, not built yet: scale deposited vigor by local maneuver difficulty (`Vdep = Vagent * (1.0 + B * |steer|)`) instead of uniform vigor across a whole successful path
- [ ] Push to origin: hold until `Documentation/rough_draft.md` has been read. Nothing has been pushed; the repo has commits `8200c19`, `dc49fa1`, `2b56049`, `059ab93`, `4822987`

## Separate, parked conversations
- [ ] Decide what to actually call the overall algorithm (SAC + pheromone reward-shaping — functional description settled, no name picked yet)
- [ ] Main SAC project (traffic-simulator) — separate from this, last known state: resumed training past 850k steps, steering jitter above ~15 m/s still unresolved, README/blog Part 5+ still pending