"""
Multi-bee training (Phase 4). Kaggle-ready: CPU workers, periodic checkpoints,
detailed CSV/JSON outputs so the run can be analysed after the session ends.

Layout
  learner (this process): owns the shared SAC policy, the replay buffer, the
                          pheromone table, and the reputation tracker. It is the
                          only writer of all of them.
  workers (N processes):  each runs its own MetaDrive env and a CPU actor copy.
                          They send transitions and episode summaries up and never
                          touch the buffer or the table.

Messages
  worker -> learner   result_q:       ("transitions", id, [(s, a, r, s', done), ...])
                                      ("episode", id, summary dict)
                                      ("error", id, traceback text)
                                      ("done", id, None)
  learner -> worker   control_q[id]:  ("weights", actor state_dict)
                                      ("table", PheromoneTableV2 snapshot)
                                      None (stop)

Outputs (in <out-dir>/<run-name>/)
  run_config.json      every setting used (including the deviations below)
  log.txt              everything printed
  episodes.csv         one row per episode: bee, caste, E, vigor, terminal, steps, path, return, reading flag
  updates.csv          losses / entropy / alpha every 500 updates
  checkpoint_<N>k/     actor, critic, alpha, pheromone table at every --ckpt-every steps
  final/               same, at the end
  summary.json         first arrival, per-window arrival rates and terminal counts, per-bee E and caste

Deviations from the frozen single-bee settings (disclose when comparing):
  buffer 50k (frozen 200k); table sent to workers every K episodes, not every episode;
  3 workers share one policy; ShapingCalculator (eta weights), not shaping.py (beta).

Stall rescue (optional, --stall-rescue; see stall_rescue.py)
  Learner-side. After a stall is flagged (default: 20 post-warmup episodes in a row that all
  time out) it adds a forward-distance bonus, an action-diversity penalty across bees, and a
  state-novelty bonus to the transition rewards. NOT potential-based; off by default. An arm
  that uses it needs its own no-shaping control (same flags plus --read-start 999999).
  episodes.csv gains stall_active, forward_sum, diversity_sum, explore_sum, bonus_total,
  max_abs_step_bonus (all zero when the rescue is off).

Usage
  python multi_bee_training.py --run-name mb_s0 --bees 3 --steps 60000 --seed 0
  python multi_bee_training.py --run-name mb_rescue_s0 --stall-rescue
"""
import argparse
import copy
import csv
import dataclasses
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import time
import traceback

import numpy as np

from config import BeeConfig, default_config
from dance_reader import DanceReader
from pheromone_table_v2 import PheromoneTableV2, decay_rate_from_half_life
from reputation_tracker import ReputationTracker
from shaping_calculator import ShapingCalculator

# Heavy imports (torch, MetaDrive) happen inside functions. Each spawned worker
# re-imports this module on Windows/Linux-spawn, and MetaDrive must not load at module level.


def env_config(config, start_seed=0, num_scenarios=1):
    """start_seed / num_scenarios pick the MetaDrive scenarios (road geometry + traffic).
    env.reset(seed=k) is only legal for start_seed <= k < start_seed + num_scenarios."""
    return {
        "start_seed": start_seed,
        "num_scenarios": num_scenarios,
        "map": config.map_type,
        "use_render": config.use_render,
        "horizon": config.horizon,
        "out_of_road_penalty": config.penalty,
        "crash_vehicle_penalty": config.penalty,
        "crash_object_penalty": config.penalty,
    }


def read_state(env, reader, shaping, step, config, decay_rate, allow_read=True):
    """Position, heading, speed, and the corridor read at the env's current state.
    allow_read=False (phase 2, unseen road) forces reading off; the table stays in memory."""
    reading = allow_read and step >= config.read_start_step
    shaping.set_reading_enabled(reading)
    pos = tuple(env.agent.position)
    heading = env.agent.heading_theta
    speed = env.agent.speed_km_h
    readout = reader.read_corridor(position=pos, heading_angle=heading, step_now=step,
                                   decay_rate=decay_rate, reading_enabled=reading)
    return pos, heading, speed, readout


def run_episode(env, actor, reader, shaping, counter, config, decay_rate, warmup,
                chunk_steps=0, flush=None, warmup_steer=1.0, warmup_throttle_min=0.3,
                allow_read=True, scenario_seed=None):
    """scenario_seed: None = plain env.reset(); k = env.reset(seed=k) (multi-scenario env).
    chunk_steps > 0: every chunk_steps transitions, hand the chunk to flush(chunk). flush returns
    True when the worker was told to stop; the episode is then abandoned and (None, None) returned.
    The returned transitions are only the unflushed remainder (all of them when chunk_steps == 0)."""
    from single_bee_training import reset_env, is_real_terminal, CRASH_FLAGS

    if scenario_seed is None:
        obs, _ = reset_env(env)
    else:
        res = env.reset(seed=scenario_seed)
        obs = res[0] if isinstance(res, tuple) else res
    start_step = counter.value
    pos0, heading, speed, readout = read_state(env, reader, shaping, start_step, config, decay_rate, allow_read)
    start_state = (pos0, heading, speed)  # state BEFORE the first action (stall rescue needs it)
    phi_s = shaping.compute_potential(heading, speed, readout)

    transitions, path, actions = [], [], []
    raw_return, shaping_abs, reward_abs = 0.0, 0.0, 0.0
    done = False
    info = {}
    while not done:
        if warmup:
            action = np.array([np.random.uniform(-warmup_steer, warmup_steer),
                               np.random.uniform(warmup_throttle_min, 1.0)])
        else:
            action, _ = actor.sample_action(obs)

        step_res = env.step(action)
        if len(step_res) == 5:
            next_obs, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_res

        # Potential at the state after the step. Zero at a real terminal.
        pos, heading, speed, readout = read_state(env, reader, shaping, counter.value, config, decay_rate, allow_read)
        path.append((pos, heading, float(action[0]), speed))
        actions.append((float(action[0]), float(action[1])))
        terminal = is_real_terminal(info)
        phi_next = 0.0 if terminal else shaping.compute_potential(heading, speed, readout)
        shaped = config.reward_scale * shaping.compute_shaped_reward(reward, phi_s, phi_next, gamma=config.gamma)
        # Only a real terminal is done=1. A horizon timeout still bootstraps.
        transitions.append((obs, action, shaped, next_obs, float(terminal)))
        if chunk_steps and len(transitions) >= chunk_steps:
            if flush(transitions):
                return None, None
            transitions = []

        raw_return += reward
        reward_abs += abs(reward)
        shaping_abs += abs(shaped - config.reward_scale * reward)
        phi_s = phi_next
        obs = next_obs

    summary = {
        "path": path,
        "actions": actions,          # [steer, throttle] per step, for the learner-side stall rescue
        "start_state": start_state,
        "arrived": bool(info.get("arrive_dest", False)),
        "crash_type": next((f for f in CRASH_FLAGS if info.get(f, False)), None),
        "steps": len(path),
        "raw_return": raw_return,
        "reading_on": bool(allow_read and start_step >= config.read_start_step),
        "route_completion": float(info.get("route_completion", 0.0)),  # MetaDrive: 0 = start, 1 = destination
        # mean |F| / mean |r| for this episode; the single-bee pilots saw 0.11-0.13
        "shaping_ratio": (shaping_abs / reward_abs) if reward_abs > 0 else 0.0,
    }
    return transitions, summary


def worker_main(worker_id, control_q, result_q, counter, produced, config, seed,
                chunk_steps=0, max_lag=0, warmup_steer=1.0, warmup_throttle_min=0.3,
                switch_step=None, phase2_seed=1, scenario_seeds=None, per_scenario=False):
    """Worker process. scenario_seeds: list of MetaDrive seeds; each episode draws one at random
    (with replacement) from this worker's own RNG. per_scenario: use that seed's own pheromone table. CPU only: hide CUDA BEFORE torch is imported anywhere in this process.

    chunk_steps > 0: send transitions every chunk_steps steps and pick up new weights between chunks
    (the learner can then train while an episode is still running). max_lag > 0: after each chunk,
    wait until the learner is within max_lag transitions of everything the workers have produced
    (backpressure, so the acting policy cannot run far ahead of the learner)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    env = None
    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(seed + worker_id)
        np.random.seed(seed + worker_id)

        from metadrive import MetaDriveEnv
        from single_bee_training import SACActor  # DEVICE resolves to cpu here

        if scenario_seeds:
            env = MetaDriveEnv(config=env_config(config, min(scenario_seeds),
                                                 max(scenario_seeds) - min(scenario_seeds) + 1))
        else:
            env = MetaDriveEnv(config=env_config(config))
        pick_rng = np.random.RandomState(seed * 1000 + worker_id + 17)
        actor = SACActor(config.obs_dim, config.action_dim, config.hidden_dim)
        reader = DanceReader(PheromoneTableV2(grid_size=config.grid_size),
                             num_lookahead_cells=config.lookahead_cells)
        shaping = ShapingCalculator(config)
        decay_rate = decay_rate_from_half_life(config.half_life_frac * config.phase1_total_steps)

        # "tables": {table key: PheromoneTableV2}; key = scenario seed if per_scenario else 0.
        state = {"version": 0, "switched": False, "tables": {}, "key": 0}  # learner update count that produced the weights in use

        def apply_control():
            """Apply every waiting control message. True = the learner told us to stop."""
            while True:
                try:
                    msg = control_q.get_nowait()
                except queue.Empty:
                    return False
                if msg is None:
                    return True
                kind, payload = msg
                if kind == "weights":
                    state["version"], weights = payload
                    actor.load_state_dict(weights)
                elif kind == "table":
                    state["tables"] = payload
                    reader.table = payload.get(state["key"]) or PheromoneTableV2(grid_size=config.grid_size)

        def send(chunk):
            if not chunk:
                return
            result_q.put(("transitions", worker_id, (state["version"], chunk)))
            with produced.get_lock():
                produced.value += len(chunk)

        def flush(chunk):
            send(chunk)
            while max_lag > 0 and produced.value - counter.value > max_lag:
                if apply_control():
                    return True
                time.sleep(0.005)
            return apply_control()

        while True:
            if apply_control():
                break
            warmup = counter.value < config.warmup_steps
            # Scenario switch happens only at an episode boundary, so an episode never mixes roads.
            in_phase2 = switch_step is not None and counter.value >= switch_step
            if in_phase2 and not state["switched"]:
                env.close()  # the engine is a singleton: close before building the next env
                env = MetaDriveEnv(config=env_config(config, phase2_seed))
                state["switched"] = True
            chosen = int(pick_rng.choice(scenario_seeds)) if scenario_seeds else None
            state["key"] = chosen if (per_scenario and chosen is not None) else 0
            reader.table = state["tables"].get(state["key"]) or PheromoneTableV2(grid_size=config.grid_size)
            transitions, summary = run_episode(env, actor, reader, shaping, counter, config,
                                               decay_rate, warmup, chunk_steps, flush,
                                               warmup_steer, warmup_throttle_min,
                                               allow_read=not in_phase2, scenario_seed=chosen)
            if summary is None:  # told to stop mid-episode
                break
            summary["phase"] = 2 if in_phase2 else 1
            summary["scenario_seed"] = chosen if chosen is not None else (phase2_seed if in_phase2 else 0)
            result_q.put(("episode", worker_id, summary))
            send(transitions)
    except Exception:
        result_q.put(("error", worker_id, traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        result_q.put(("done", worker_id, None))


def save_checkpoint(folder, agent, table, tracker, num_bees, total_steps, episodes):
    import torch
    os.makedirs(folder, exist_ok=True)
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "total_steps": total_steps,
        "episodes": episodes,
    }, os.path.join(folder, "agent.pt"))
    with open(os.path.join(folder, "pheromone_table.pkl"), "wb") as f:
        pickle.dump(table, f)
    with open(os.path.join(folder, "tracker.json"), "w") as f:
        json.dump({"E": {str(i): tracker.E[i] for i in range(num_bees)},
                   "caste": {str(i): tracker.caste(i) for i in range(num_bees)}}, f, indent=2)


def window_stats(rows, budget, window):
    """Per-window arrival rate and terminal counts, keyed by the global step at episode end."""
    out = []
    for lo in range(0, budget, window):
        w = [r for r in rows if lo <= r["global_step"] < lo + window]
        counts = {}
        for r in w:
            counts[r["terminal"]] = counts.get(r["terminal"], 0) + 1
        n = len(w)
        out.append({
            "steps": f"{lo}-{lo + window}",
            "episodes": n,
            "arrival_rate": (counts.get("arrive_dest", 0) / n) if n else None,
            "terminals": counts,
            "mean_path_m": (sum(r["path_m"] for r in w) / n) if n else None,
            "max_path_m": max((r["path_m"] for r in w), default=None),
            "mean_raw_return": (sum(r["raw_return"] for r in w) / n) if n else None,
        })
    return out


def scenario_summaries(rows):
    """Per scenario seed: episodes, arrivals, mean route completion %, crashes, timeouts (training episodes)."""
    out = {}
    for sd in sorted({r["seed"] for r in rows}):
        w = [r for r in rows if r["seed"] == sd]
        out[str(sd)] = {"episodes": len(w),
                        "arrivals": sum(1 for r in w if r["terminal"] == "arrive_dest"),
                        "first_arrival_step": min((r["global_step"] for r in w if r["terminal"] == "arrive_dest"),
                                                  default=None),
                        "avg_route_completion_pct": round(100 * sum(r["rc"] for r in w) / len(w), 1),
                        "crashes": sum(1 for r in w if r["terminal"] not in ("arrive_dest", "timeout")),
                        "timeouts": sum(1 for r in w if r["terminal"] == "timeout")}
    return out


def phase_summaries(rows, phase1_cells):
    """Per-phase arrivals, first arrival, mean distance covered (route_completion), terminal counts."""
    out = []
    for ph in (1, 2):
        w = [r for r in rows if r.get("phase", 1) == ph]
        if not w:
            continue
        arr = [r for r in w if r["terminal"] == "arrive_dest"]
        d = {"phase": ph, "episodes": len(w), "arrivals": len(arr),
             "first_arrival_step": min((r["global_step"] for r in arr), default=None),
             "avg_distance_covered_pct": round(100 * sum(r["rc"] for r in w) / len(w), 1),
             "crashes": sum(1 for r in w if r["terminal"] not in ("arrive_dest", "timeout")),
             "timeouts": sum(1 for r in w if r["terminal"] == "timeout")}
        if ph == 1:
            d["table_cells_at_switch"] = phase1_cells if phase1_cells is not None else "n/a"
        out.append(d)
    return out


def run_multi_bee(config=default_config, num_bees=3, run_dir="multi_bee_run", seed=0,
                  sync_every_updates=200, table_every_episodes=10, ckpt_every=10000,
                  window=10000, rescue_cfg=None, buffer_capacity=50000, chunk_steps=0, max_lag=0,
                  warmup_steer=1.0, warmup_throttle_min=0.3, novice_writes=False,
                  novice_success_vigor=None, switch_step=None, phase2_seed=1,
                  scenario_seeds=None, per_scenario=False):
    """switch_step: None = single scenario (old behaviour). N = once N transitions are in, each worker
    rebuilds its env on start_seed=phase2_seed at its next episode boundary, with reading off and no
    table writes. Learning continues in phase 2 (adaptation, not a frozen zero-shot test).
    rescue_cfg: None = plain run. A dict of StallRescue kwargs turns the stall rescue on.
    chunk_steps / max_lag: see worker_main. Chunking is incompatible with the rescue (the rescue
    scores a whole episode before its transitions are stored)."""
    if scenario_seeds and switch_step is not None:
        raise SystemExit("--scenario-seeds cannot be combined with --switch-step")
    if per_scenario and not scenario_seeds:
        raise SystemExit("--pheromone-per-scenario needs --scenario-seeds")
    if chunk_steps and rescue_cfg is not None:
        raise SystemExit("--chunk-steps cannot be combined with --stall-rescue")
    from single_bee_training import SACAgent, ReplayBuffer, write_success_path, write_crash_point
    from stall_rescue import StallRescue
    import torch

    os.makedirs(run_dir, exist_ok=True)
    if os.listdir(run_dir):
        raise SystemExit(f"{run_dir} is not empty. Pick a new --run-name.")
    log_f = open(os.path.join(run_dir, "log.txt"), "a", buffering=1)

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")

    np.random.seed(seed)
    torch.manual_seed(seed)

    decay_rate = decay_rate_from_half_life(config.half_life_frac * config.phase1_total_steps)
    budget = config.phase1_total_steps

    run_cfg = dataclasses.asdict(config)
    run_cfg.update({"num_bees": num_bees, "seed": seed, "buffer_capacity_used": buffer_capacity,
                    "chunk_steps": chunk_steps, "max_lag": max_lag,
                    "warmup_steer": warmup_steer, "warmup_throttle_min": warmup_throttle_min,
                    "novice_writes": novice_writes, "novice_success_vigor": novice_success_vigor,
                    "eta_success": config.eta_success,
                    "switch_step": switch_step, "phase2_seed": phase2_seed,
                    "scenario_seeds": scenario_seeds, "pheromone_per_scenario": per_scenario,
                    "distance_covered_metric": "MetaDrive info['route_completion'] at episode end",
                    "sync_every_updates": sync_every_updates,
                    "sync_every_updates": sync_every_updates,
                    "table_every_episodes": table_every_episodes,
                    "budget_steps": budget, "decay_rate": decay_rate,
                    "cuda_in_learner": bool(torch.cuda.is_available()),
                    "stall_rescue": rescue_cfg})
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    ctx = mp.get_context("spawn")
    counter = ctx.Value("l", 0)  # learner's transition count; workers read it for the reading gate
    produced = ctx.Value("l", 0)  # transitions the workers have sent (for the max_lag backpressure)
    result_q = ctx.Queue()
    control_qs = [ctx.Queue() for _ in range(num_bees)]
    workers = [ctx.Process(target=worker_main,
                           args=(i, control_qs[i], result_q, counter, produced, config, seed,
                                 chunk_steps, max_lag, warmup_steer, warmup_throttle_min,
                                 switch_step, phase2_seed, scenario_seeds, per_scenario), daemon=True)
               for i in range(num_bees)]

    agent = SACAgent(config)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    # {table key: PheromoneTableV2}. Key = scenario seed with --pheromone-per-scenario, else 0 (one global table).
    tables = {}

    def get_table(key):
        if key not in tables:
            tables[key] = PheromoneTableV2(grid_size=config.grid_size)
        return tables[key]

    def n_cells():
        return sum(len(t.table) for t in tables.values())

    def ckpt_table():
        """Checkpoint format: a dict {seed: table} when per-scenario, else the single table (old format)."""
        return tables if per_scenario else get_table(0)

    if not per_scenario:
        get_table(0)  # old single-table behaviour: the table exists from the start
    tracker = ReputationTracker(agent_ids=list(range(num_bees)))
    rescue = (StallRescue(num_bees, warmup_steps=config.warmup_steps, **rescue_cfg)
              if rescue_cfg is not None else None)
    pending_bonus = {}  # worker_id -> per-step bonuses for the episode whose transitions are next

    def broadcast_weights(version):
        """version = learner update count; workers tag their transitions with it so we can log lag."""
        state = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
        for q in control_qs:
            q.put(("weights", (version, state)))

    def broadcast_table():
        snapshot = copy.deepcopy(tables)  # a fresh copy, so queue pickling never races a write
        for q in control_qs:
            q.put(("table", snapshot))

    log("=" * 80)
    log(f"MULTI-BEE | {num_bees} workers | budget {budget} steps | seed {seed} | run dir {run_dir}")
    log(f"warmup {config.warmup_steps} | reading from step {config.read_start_step} | "
        f"half-life {config.half_life_frac * budget:.0f} steps | penalty {config.penalty} | "
        f"horizon {config.horizon} | learner device cuda={torch.cuda.is_available()}")
    log(f"stall rescue: {'OFF' if rescue is None else rescue_cfg}")
    log("=" * 80)

    ep_f = open(os.path.join(run_dir, "episodes.csv"), "w", newline="")
    ep_w = csv.writer(ep_f)
    ep_w.writerow(["episode", "bee", "caste", "E_before", "vigor", "terminal", "steps", "path_m",
                   "raw_return", "global_step", "elapsed_s", "reading_on", "shaping_ratio",
                   "table_cells", "wrote", "stall_active", "forward_sum", "diversity_sum",
                   "explore_sum", "bonus_total", "max_abs_step_bonus", "phase", "scenario_seed",
                   "route_completion"])
    up_f = open(os.path.join(run_dir, "updates.csv"), "w", newline="")
    up_w = csv.writer(up_f)
    up_w.writerow(["update", "global_step", "elapsed_s", "critic_loss", "actor_loss", "entropy",
                   "alpha", "buffer", "mean_lag"])

    rows = []
    first_arrival = None
    start_time = time.time()
    for p in workers:
        p.start()
    broadcast_weights(0)
    broadcast_table()

    total_steps, updates, episodes, active = 0, 0, 0, num_bees
    lag_sum, lag_n = 0, 0  # policy lag in updates: learner's update count minus the weights' version
    next_ckpt = ckpt_every
    switch_logged = False
    phase1_cells = None
    errors = []
    try:
        while active > 0 and total_steps < budget:
            try:
                kind, worker_id, payload = result_q.get(timeout=1.0)
            except queue.Empty:
                if not any(p.is_alive() for p in workers):
                    log("  [WARNING] all workers dead")
                    break
                continue

            if kind == "error":
                errors.append((worker_id, payload))
                log(f"\n  [BEE {worker_id}] CRASHED\n{payload}\n")

            elif kind == "done":
                active -= 1
                log(f"  [BEE {worker_id}] finished | {active} workers remain")

            elif kind == "transitions":
                version, payload = payload
                lag_sum += (updates - version) * len(payload)
                lag_n += len(payload)
                bonus = pending_bonus.pop(worker_id, None)  # this worker's episode message came first
                if bonus is not None:
                    assert len(bonus) == len(payload), (len(bonus), len(payload))
                    payload = [(o, a, r + float(b), no, d)
                               for (o, a, r, no, d), b in zip(payload, bonus)]
                for t in payload:
                    buffer.push(*t)
                total_steps += len(payload)
                counter.value = total_steps
                if switch_step is not None and not switch_logged and total_steps >= switch_step:
                    switch_logged = True
                    phase1_cells = n_cells()
                    log(f"\n{'#' * 80}\nStep {total_steps}: switching scenario 0 -> scenario {phase2_seed}, "
                        f"pheromone reading OFF (table kept, {phase1_cells} cells, no phase-2 writes)\n{'#' * 80}")
                    save_checkpoint(os.path.join(run_dir, "checkpoint_phase1_end"),
                                    agent, ckpt_table(), tracker, num_bees, total_steps, episodes)

                for _ in range(len(payload)):  # one update per transition, as in single-bee
                    if len(buffer) < config.warmup_steps:
                        break
                    critic_loss, actor_loss, alpha_loss, entropy, alpha = agent.update_parameters(
                        buffer, config.batch_size)
                    if critic_loss is None:
                        break
                    updates += 1
                    if updates % sync_every_updates == 0:
                        broadcast_weights(updates)
                    if updates % 500 == 0:
                        el = time.time() - start_time
                        mean_lag = lag_sum / lag_n if lag_n else 0.0
                        lag_sum, lag_n = 0, 0
                        up_w.writerow([updates, total_steps, f"{el:.0f}", f"{critic_loss:.4f}",
                                       f"{actor_loss:.4f}", f"{entropy:.4f}", f"{alpha:.4f}",
                                       len(buffer), f"{mean_lag:.0f}"])
                        up_f.flush()
                        log(f"    [Update {updates:6d} | step {total_steps:6d} | {el:5.0f}s] "
                            f"critic={critic_loss:8.2f} actor={actor_loss:8.4f} "
                            f"entropy={entropy:6.3f} alpha={alpha:.3f} lag={mean_lag:.0f}")

                if total_steps >= next_ckpt:
                    save_checkpoint(os.path.join(run_dir, f"checkpoint_{next_ckpt // 1000}k"),
                                    agent, ckpt_table(), tracker, num_bees, total_steps, episodes)
                    log(f"  -- checkpoint saved at step {total_steps}")
                    next_ckpt += ckpt_every

            elif kind == "episode":
                episodes += 1
                arrived = payload["arrived"]
                crash_type = payload["crash_type"]
                path = payload["path"]
                caste = tracker.caste(worker_id)  # the caste this bee drove with
                vigor = tracker.vigor(worker_id)
                e_before = tracker.E[worker_id]

                phase = payload.get("phase", 1)
                rc = payload.get("route_completion", 0.0)
                sc_seed = payload.get("scenario_seed", 0)
                wkey = sc_seed if per_scenario else 0  # which pheromone table this episode writes to
                wrote = "none"
                novice_ok = novice_writes or novice_success_vigor is not None
                if phase == 2:
                    pass  # unseen road: its positions mean nothing in the scenario-0 table
                elif arrived and (caste != "NOVICE" or novice_ok):
                    if caste == "NOVICE" and novice_success_vigor is not None:
                        vigor = novice_success_vigor  # stronger trace for a NOVICE arrival
                    write_success_path(get_table(wkey), path, vigor, total_steps, decay_rate)
                    wrote = "success"
                elif crash_type is not None:
                    write_crash_point(get_table(wkey), path, crash_type, vigor, total_steps, decay_rate)
                    wrote = "crash"
                tracker.update(worker_id, success=arrived)

                if episodes % table_every_episodes == 0 or arrived:  # arrivals are rare: push at once
                    broadcast_table()

                path_m = sum(math.hypot(b[0][0] - a[0][0], b[0][1] - a[0][1])
                             for a, b in zip(path, path[1:]))
                terminal = "arrive_dest" if arrived else (crash_type or "timeout")
                el = time.time() - start_time

                # Stall rescue: decide the flag from this episode's outcome, then score the episode.
                diag = {"forward_sum": 0.0, "diversity_sum": 0.0, "explore_sum": 0.0,
                        "total_sum": 0.0, "max_abs_step": 0.0}
                if rescue is not None:
                    if rescue.observe_episode(terminal, total_steps):
                        log(f"  *** STALL FLAGGED at step {total_steps} (episode {episodes}): "
                            f"last {rescue.stall_window} post-warmup episodes were "
                            f"{'all timeouts' if rescue.stall_mode == 'timeout' else 'without an arrival'}. "
                            f"Curriculum bonuses are ON from here (latched).")
                    bonuses, diag = rescue.episode_bonuses(worker_id, payload["start_state"], path,
                                                           payload["actions"])
                    pending_bonus[worker_id] = bonuses
                    if episodes % 10 == 0:
                        log(f"  [stall check] ep {episodes} | active={rescue.active} | recent="
                            f"{len(rescue.recent)}/{rescue.stall_window} | timeouts in window="
                            f"{sum(1 for t in rescue.recent if t == 'timeout')} | last ep bonus "
                            f"fwd={diag['forward_sum']:.2f} div={diag['diversity_sum']:.2f} "
                            f"exp={diag['explore_sum']:.2f} max|step|={diag['max_abs_step']:.2f}")
                if arrived and first_arrival is None:
                    first_arrival = {"episode": episodes, "bee": worker_id, "global_step": total_steps}
                    log(f"  *** FIRST ARRIVAL: episode {episodes}, bee {worker_id}, step {total_steps}")

                rows.append({"global_step": total_steps, "terminal": terminal, "path_m": path_m,
                             "raw_return": payload["raw_return"], "phase": phase, "rc": rc,
                             "seed": sc_seed})
                ep_w.writerow([episodes, worker_id, caste, f"{e_before:.4f}", f"{vigor:.3f}", terminal,
                               payload["steps"], f"{path_m:.2f}", f"{payload['raw_return']:.2f}",
                               total_steps, f"{el:.0f}", int(payload["reading_on"]),
                               f"{payload['shaping_ratio']:.3f}", n_cells(), wrote,
                               int(rescue.active) if rescue is not None else 0,
                               f"{diag['forward_sum']:.3f}", f"{diag['diversity_sum']:.3f}",
                               f"{diag['explore_sum']:.3f}", f"{diag['total_sum']:.3f}",
                               f"{diag['max_abs_step']:.3f}", phase,
                               payload.get("scenario_seed", 0), f"{rc:.4f}"])
                ep_f.flush()
                log(f"  Ep {episodes:4d} | bee {worker_id} | {caste:<7} E={e_before:.2f} | "
                    f"{terminal:<14} | steps={payload['steps']:3d} | path={path_m:6.1f}m | "
                    f"R={payload['raw_return']:7.1f} | P{phase} distance_covered: {rc * 100:.0f}% | "
                    f"arrival: {arrived} | read={int(payload['reading_on'])} "
                    f"F/r={payload['shaping_ratio']:.2f} | cells={n_cells()} | step={total_steps}")
    finally:
        log("\nShutting down workers...")
        for q in control_qs:
            q.put(None)
        # Drain the result queue so workers can exit instead of blocking on a full pipe.
        deadline = time.time() + 60
        while active > 0 and time.time() < deadline:
            try:
                kind, worker_id, payload = result_q.get(timeout=1.0)
            except queue.Empty:
                if not any(p.is_alive() for p in workers):
                    break
                continue
            if kind == "done":
                active -= 1
            elif kind == "error":
                errors.append((worker_id, payload))
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

        # Always save, even after an error or Ctrl-C.
        save_checkpoint(os.path.join(run_dir, "final"), agent, ckpt_table(), tracker, num_bees,
                        total_steps, episodes)
        elapsed = time.time() - start_time
        summary = {
            "elapsed_s": elapsed,
            "steps_per_s": total_steps / elapsed if elapsed > 0 else 0.0,
            "transitions": total_steps, "updates": updates, "episodes": episodes,
            "first_arrival": first_arrival,
            "arrivals": sum(1 for r in rows if r["terminal"] == "arrive_dest"),
            "terminal_counts": {t: sum(1 for r in rows if r["terminal"] == t)
                                for t in sorted({r["terminal"] for r in rows})},
            "windows": window_stats(rows, budget, window),
            "per_bee": {str(i): {"E": tracker.E[i], "caste": tracker.caste(i)} for i in range(num_bees)},
            "pheromone_cells": n_cells(),
            "stall_rescue": {"enabled": rescue is not None, "config": rescue_cfg,
                             "activated_at_step": rescue.activated_at if rescue is not None else None},
            "phases": phase_summaries(rows, phase1_cells),
            "scenarios": scenario_summaries(rows) if scenario_seeds else None,
            "worker_errors":[{"bee": w, "traceback": tb} for w, tb in errors],
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        ep_f.close()
        up_f.close()

    log(f"\nRun finished in {elapsed:.0f}s ({summary['steps_per_s']:.1f} steps/s)")
    log(f"Transitions {total_steps} | updates {updates} | episodes {episodes} | "
        f"arrivals {summary['arrivals']} | first arrival {first_arrival}")
    log(f"Terminals: {summary['terminal_counts']}")
    log(f"Castes: {[tracker.caste(i) for i in range(num_bees)]} | table cells {n_cells()}")
    for w in summary["windows"]:
        rate = "n/a" if w["arrival_rate"] is None else f"{w['arrival_rate']:.2f}"
        log(f"  steps {w['steps']:>13}: {w['episodes']:3d} eps | arrival {rate} | "
            f"max path {w['max_path_m'] or 0:6.1f} m | {w['terminals']}")
    if switch_step is not None:
        log(f"\nPHASE SUMMARY | run {os.path.basename(run_dir)}")
        for ph in summary["phases"]:
            log("  " + "  |  ".join(f"{k}={v}" for k, v in ph.items()))
    log_f.close()
    return agent, ckpt_table(), tracker


def default_out_root():
    return "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", "multi_bee")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-root", default=default_out_root())
    ap.add_argument("--bees", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60000, help="total transition budget")
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--novice-writes", action="store_true",
                    help="NOVICE arrivals also write success traces (at NOVICE vigor 0.05). Default off = option A.")
    ap.add_argument("--novice-success-vigor", type=float, default=None,
                    help="NOVICE arrivals write success traces at this vigor (implies --novice-writes). "
                         "Default None = off, or 0.05 with --novice-writes.")
    ap.add_argument("--eta-success", type=float, default=None,
                    help="weight of the success term in the shaping potential (default 0.5 from config.py)")
    ap.add_argument("--read-start", type=int, default=None, help="default = end of warmup")
    ap.add_argument("--half-life-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--table-every", type=int, default=10, help="episodes between table broadcasts")
    ap.add_argument("--buffer", type=int, default=50000,
                    help="replay buffer capacity (frozen single-bee setting is 200000)")
    ap.add_argument("--chunk-steps", type=int, default=0,
                    help="0 = send a whole episode at the end (old behaviour). N > 0 = send every N "
                         "steps and pick up new weights between chunks. Not usable with --stall-rescue.")
    ap.add_argument("--max-lag", type=int, default=0,
                    help="0 = workers run freely. N > 0 = a worker waits while more than N transitions "
                         "are sent but not yet received by the learner (needs --chunk-steps).")
    ap.add_argument("--warmup-steer", type=float, default=1.0,
                    help="warmup steering is uniform(-X, X). Default 1.0 (the multi-bee runs so far). "
                         "The single-bee loop uses 0.3.")
    ap.add_argument("--warmup-throttle-min", type=float, default=0.3,
                    help="warmup throttle is uniform(X, 1.0). Default 0.3. Single-bee loop uses 0.2.")
    ap.add_argument("--switch-step", type=int, default=None,
                    help="at this transition count, workers move to --phase2-seed with reading OFF "
                         "(default None = single scenario, old behaviour)")
    ap.add_argument("--phase2-seed", type=int, default=1,
                    help="MetaDrive start_seed (scenario) for phase 2. Seed 1 route is 280 m vs 241 m.")
    ap.add_argument("--scenario-seeds", type=str, default=None,
                    help="comma list of MetaDrive scenario seeds, e.g. 0,1,2,3,4. Each episode draws one at "
                         "random (with replacement). Default None = scenario 0 only.")
    ap.add_argument("--pheromone-per-scenario", action="store_true",
                    help="one pheromone table per scenario seed (needs --scenario-seeds). "
                         "Without it all scenarios share one table.")
    ap.add_argument("--sync-every", type=int, default=None,
                    help="learner updates between weight broadcasts (default 200, or 25 with chunking)")
    # Stall rescue (off unless --stall-rescue). Defaults come from stall_rescue.StallRescue.
    ap.add_argument("--stall-rescue", action="store_true",
                    help="turn on the adaptive curriculum + diversity terms after a stall is flagged")
    ap.add_argument("--stall-window", type=int, default=20)
    ap.add_argument("--stall-mode", choices=["timeout", "arrival"], default="timeout")
    ap.add_argument("--forward-scale", type=float, default=0.5)
    ap.add_argument("--diversity-scale", type=float, default=0.005)
    ap.add_argument("--explore-scale", type=float, default=0.05)
    ap.add_argument("--sim-threshold", type=float, default=0.90)
    args = ap.parse_args()
    rescue_cfg = None
    if args.stall_rescue:
        rescue_cfg = {"stall_window": args.stall_window, "stall_mode": args.stall_mode,
                      "forward_scale": args.forward_scale, "diversity_scale": args.diversity_scale,
                      "explore_scale": args.explore_scale, "sim_threshold": args.sim_threshold}

    cfg = dataclasses.replace(
        default_config,
        phase1_total_steps=args.steps,
        warmup_steps=args.warmup,
        read_start_step=args.read_start if args.read_start is not None else args.warmup,
        half_life_frac=args.half_life_frac,
        **({"eta_success": args.eta_success} if args.eta_success is not None else {}),
    )
    run_multi_bee(cfg, num_bees=args.bees, run_dir=os.path.join(args.out_root, args.run_name),
                  seed=args.seed, ckpt_every=args.ckpt_every, table_every_episodes=args.table_every,
                  rescue_cfg=rescue_cfg, buffer_capacity=args.buffer,
                  chunk_steps=args.chunk_steps, max_lag=args.max_lag,
                  warmup_steer=args.warmup_steer, warmup_throttle_min=args.warmup_throttle_min,
                  novice_writes=args.novice_writes, novice_success_vigor=args.novice_success_vigor,
                  switch_step=args.switch_step, phase2_seed=args.phase2_seed,
                  scenario_seeds=([int(x) for x in args.scenario_seeds.split(",")] if args.scenario_seeds else None),
                  per_scenario=args.pheromone_per_scenario,
                  sync_every_updates=(args.sync_every if args.sync_every is not None
                                      else (25 if args.chunk_steps else 200)))
