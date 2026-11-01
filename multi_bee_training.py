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


def env_config(config):
    return {
        "map": config.map_type,
        "use_render": config.use_render,
        "horizon": config.horizon,
        "out_of_road_penalty": config.penalty,
        "crash_vehicle_penalty": config.penalty,
        "crash_object_penalty": config.penalty,
    }


def read_state(env, reader, shaping, step, config, decay_rate):
    """Position, heading, speed, and the corridor read at the env's current state."""
    reading = step >= config.read_start_step
    shaping.set_reading_enabled(reading)
    pos = tuple(env.agent.position)
    heading = env.agent.heading_theta
    speed = env.agent.speed_km_h
    readout = reader.read_corridor(position=pos, heading_angle=heading, step_now=step,
                                   decay_rate=decay_rate, reading_enabled=reading)
    return pos, heading, speed, readout


def run_episode(env, actor, reader, shaping, counter, config, decay_rate, warmup):
    from single_bee_training import reset_env, is_real_terminal, CRASH_FLAGS

    obs, _ = reset_env(env)
    start_step = counter.value
    pos0, heading, speed, readout = read_state(env, reader, shaping, start_step, config, decay_rate)
    start_state = (pos0, heading, speed)  # state BEFORE the first action (stall rescue needs it)
    phi_s = shaping.compute_potential(heading, speed, readout)

    transitions, path, actions = [], [], []
    raw_return, shaping_abs, reward_abs = 0.0, 0.0, 0.0
    done = False
    info = {}
    while not done:
        if warmup:
            action = np.array([np.random.uniform(-1, 1), np.random.uniform(0.3, 1.0)])
        else:
            action, _ = actor.sample_action(obs)

        step_res = env.step(action)
        if len(step_res) == 5:
            next_obs, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_res

        # Potential at the state after the step. Zero at a real terminal.
        pos, heading, speed, readout = read_state(env, reader, shaping, counter.value, config, decay_rate)
        path.append((pos, heading, float(action[0]), speed))
        actions.append((float(action[0]), float(action[1])))
        terminal = is_real_terminal(info)
        phi_next = 0.0 if terminal else shaping.compute_potential(heading, speed, readout)
        shaped = config.reward_scale * shaping.compute_shaped_reward(reward, phi_s, phi_next, gamma=config.gamma)
        # Only a real terminal is done=1. A horizon timeout still bootstraps.
        transitions.append((obs, action, shaped, next_obs, float(terminal)))

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
        "reading_on": bool(start_step >= config.read_start_step),
        # mean |F| / mean |r| for this episode; the single-bee pilots saw 0.11-0.13
        "shaping_ratio": (shaping_abs / reward_abs) if reward_abs > 0 else 0.0,
    }
    return transitions, summary


def worker_main(worker_id, control_q, result_q, counter, config, seed):
    """Worker process. CPU only: hide CUDA BEFORE torch is imported anywhere in this process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    env = None
    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(seed + worker_id)
        np.random.seed(seed + worker_id)

        from metadrive import MetaDriveEnv
        from single_bee_training import SACActor  # DEVICE resolves to cpu here

        env = MetaDriveEnv(config=env_config(config))
        actor = SACActor(config.obs_dim, config.action_dim, config.hidden_dim)
        reader = DanceReader(PheromoneTableV2(grid_size=config.grid_size),
                             num_lookahead_cells=config.lookahead_cells)
        shaping = ShapingCalculator(config)
        decay_rate = decay_rate_from_half_life(config.half_life_frac * config.phase1_total_steps)

        while True:
            stop = False
            while True:  # apply the newest control messages, stop on None
                try:
                    msg = control_q.get_nowait()
                except queue.Empty:
                    break
                if msg is None:
                    stop = True
                    break
                kind, payload = msg
                if kind == "weights":
                    actor.load_state_dict(payload)
                elif kind == "table":
                    reader.table = payload
            if stop:
                break

            warmup = counter.value < config.warmup_steps
            transitions, summary = run_episode(env, actor, reader, shaping, counter, config,
                                               decay_rate, warmup)
            result_q.put(("episode", worker_id, summary))
            result_q.put(("transitions", worker_id, transitions))
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


def run_multi_bee(config=default_config, num_bees=3, run_dir="multi_bee_run", seed=0,
                  sync_every_updates=200, table_every_episodes=10, ckpt_every=10000,
                  window=10000, rescue_cfg=None):
    """rescue_cfg: None = plain run. A dict of StallRescue kwargs turns the stall rescue on."""
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
    run_cfg.update({"num_bees": num_bees, "seed": seed, "buffer_capacity_used": 50000,
                    "sync_every_updates": sync_every_updates,
                    "table_every_episodes": table_every_episodes,
                    "budget_steps": budget, "decay_rate": decay_rate,
                    "cuda_in_learner": bool(torch.cuda.is_available()),
                    "stall_rescue": rescue_cfg})
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    ctx = mp.get_context("spawn")
    counter = ctx.Value("l", 0)  # learner's transition count; workers read it for the reading gate
    result_q = ctx.Queue()
    control_qs = [ctx.Queue() for _ in range(num_bees)]
    workers = [ctx.Process(target=worker_main,
                           args=(i, control_qs[i], result_q, counter, config, seed), daemon=True)
               for i in range(num_bees)]

    agent = SACAgent(config)
    buffer = ReplayBuffer(capacity=50000)
    table = PheromoneTableV2(grid_size=config.grid_size)
    tracker = ReputationTracker(agent_ids=list(range(num_bees)))
    rescue = (StallRescue(num_bees, warmup_steps=config.warmup_steps, **rescue_cfg)
              if rescue_cfg is not None else None)
    pending_bonus = {}  # worker_id -> per-step bonuses for the episode whose transitions are next

    def broadcast_weights():
        state = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
        for q in control_qs:
            q.put(("weights", state))

    def broadcast_table():
        snapshot = copy.deepcopy(table)  # a fresh copy, so queue pickling never races a write
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
                   "explore_sum", "bonus_total", "max_abs_step_bonus"])
    up_f = open(os.path.join(run_dir, "updates.csv"), "w", newline="")
    up_w = csv.writer(up_f)
    up_w.writerow(["update", "global_step", "elapsed_s", "critic_loss", "actor_loss", "entropy",
                   "alpha", "buffer"])

    rows = []
    first_arrival = None
    start_time = time.time()
    for p in workers:
        p.start()
    broadcast_weights()
    broadcast_table()

    total_steps, updates, episodes, active = 0, 0, 0, num_bees
    next_ckpt = ckpt_every
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
                bonus = pending_bonus.pop(worker_id, None)  # this worker's episode message came first
                if bonus is not None:
                    assert len(bonus) == len(payload), (len(bonus), len(payload))
                    payload = [(o, a, r + float(b), no, d)
                               for (o, a, r, no, d), b in zip(payload, bonus)]
                for t in payload:
                    buffer.push(*t)
                total_steps += len(payload)
                counter.value = total_steps

                for _ in range(len(payload)):  # one update per transition, as in single-bee
                    if len(buffer) < config.warmup_steps:
                        break
                    critic_loss, actor_loss, alpha_loss, entropy, alpha = agent.update_parameters(
                        buffer, config.batch_size)
                    if critic_loss is None:
                        break
                    updates += 1
                    if updates % sync_every_updates == 0:
                        broadcast_weights()
                    if updates % 500 == 0:
                        el = time.time() - start_time
                        up_w.writerow([updates, total_steps, f"{el:.0f}", f"{critic_loss:.4f}",
                                       f"{actor_loss:.4f}", f"{entropy:.4f}", f"{alpha:.4f}",
                                       len(buffer)])
                        up_f.flush()
                        log(f"    [Update {updates:6d} | step {total_steps:6d} | {el:5.0f}s] "
                            f"critic={critic_loss:8.2f} actor={actor_loss:8.4f} "
                            f"entropy={entropy:6.3f} alpha={alpha:.3f}")

                if total_steps >= next_ckpt:
                    save_checkpoint(os.path.join(run_dir, f"checkpoint_{next_ckpt // 1000}k"),
                                    agent, table, tracker, num_bees, total_steps, episodes)
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

                wrote = "none"
                if arrived and caste != "NOVICE":
                    write_success_path(table, path, vigor, total_steps, decay_rate)
                    wrote = "success"
                elif crash_type is not None:
                    write_crash_point(table, path, crash_type, vigor, total_steps, decay_rate)
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
                             "raw_return": payload["raw_return"]})
                ep_w.writerow([episodes, worker_id, caste, f"{e_before:.4f}", f"{vigor:.3f}", terminal,
                               payload["steps"], f"{path_m:.2f}", f"{payload['raw_return']:.2f}",
                               total_steps, f"{el:.0f}", int(payload["reading_on"]),
                               f"{payload['shaping_ratio']:.3f}", len(table.table), wrote,
                               int(rescue.active) if rescue is not None else 0,
                               f"{diag['forward_sum']:.3f}", f"{diag['diversity_sum']:.3f}",
                               f"{diag['explore_sum']:.3f}", f"{diag['total_sum']:.3f}",
                               f"{diag['max_abs_step']:.3f}"])
                ep_f.flush()
                log(f"  Ep {episodes:4d} | bee {worker_id} | {caste:<7} E={e_before:.2f} | "
                    f"{terminal:<14} | steps={payload['steps']:3d} | path={path_m:6.1f}m | "
                    f"R={payload['raw_return']:7.1f} | read={int(payload['reading_on'])} "
                    f"F/r={payload['shaping_ratio']:.2f} | cells={len(table.table)} | step={total_steps}")
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
        save_checkpoint(os.path.join(run_dir, "final"), agent, table, tracker, num_bees,
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
            "pheromone_cells": len(table.table),
            "stall_rescue": {"enabled": rescue is not None, "config": rescue_cfg,
                             "activated_at_step": rescue.activated_at if rescue is not None else None},
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
    log(f"Castes: {[tracker.caste(i) for i in range(num_bees)]} | table cells {len(table.table)}")
    for w in summary["windows"]:
        rate = "n/a" if w["arrival_rate"] is None else f"{w['arrival_rate']:.2f}"
        log(f"  steps {w['steps']:>13}: {w['episodes']:3d} eps | arrival {rate} | "
            f"max path {w['max_path_m'] or 0:6.1f} m | {w['terminals']}")
    log_f.close()
    return agent, table, tracker


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
    ap.add_argument("--read-start", type=int, default=None, help="default = end of warmup")
    ap.add_argument("--half-life-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--table-every", type=int, default=10, help="episodes between table broadcasts")
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
    )
    run_multi_bee(cfg, num_bees=args.bees, run_dir=os.path.join(args.out_root, args.run_name),
                  seed=args.seed, ckpt_every=args.ckpt_every, table_every_episodes=args.table_every,
                  rescue_cfg=rescue_cfg)
