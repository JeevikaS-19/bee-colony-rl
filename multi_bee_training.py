"""
Multi-bee training (Phase 4). Replaces the pseudo code in multi_bee.py.

Layout
  learner (this process): owns the shared SAC policy, the replay buffer, the
                          pheromone table, and the reputation tracker. It is the
                          only writer of all of them.
  workers (N processes):  each runs its own MetaDrive env and a CPU/device copy of
                          the actor. They send transitions and episode summaries up
                          and never touch the buffer or the table.

Messages
  worker -> learner   result_q:       ("transitions", id, [(s, a, r, s', done), ...])
                                      ("episode", id, summary dict)
                                      ("error", id, traceback text)
                                      ("done", id, None)
  learner -> worker   control_q[id]:  ("weights", actor state_dict)
                                      ("table", PheromoneTableV2 snapshot)
                                      None (stop)
"""
import copy
import math
import multiprocessing as mp
import queue
import traceback

import numpy as np

from config import BeeConfig, default_config
from dance_reader import DanceReader
from pheromone_table_v2 import PheromoneTableV2, decay_rate_from_half_life
from reputation_tracker import ReputationTracker
from shaping_calculator import ShapingCalculator

# Heavy imports (torch, MetaDrive) happen inside functions. Each spawned worker
# re-imports this module on Windows, and MetaDrive must not load at module level.


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
    _, heading, speed, readout = read_state(env, reader, shaping, counter.value, config, decay_rate)
    phi_s = shaping.compute_potential(heading, speed, readout)

    transitions, path = [], []
    raw_return = 0.0
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
        terminal = is_real_terminal(info)
        phi_next = 0.0 if terminal else shaping.compute_potential(heading, speed, readout)
        shaped = config.reward_scale * shaping.compute_shaped_reward(reward, phi_s, phi_next, gamma=config.gamma)
        transitions.append((obs, action, shaped, next_obs, float(terminal)))

        raw_return += reward
        phi_s = phi_next
        obs = next_obs

    summary = {
        "path": path,
        "arrived": bool(info.get("arrive_dest", False)),
        "crash_type": next((f for f in CRASH_FLAGS if info.get(f, False)), None),
        "steps": len(path),
        "raw_return": raw_return,
    }
    return transitions, summary


def worker_main(worker_id, control_q, result_q, counter, config):
    env = None
    try:
        from metadrive import MetaDriveEnv
        import torch
        torch.set_num_threads(1)
        from single_bee_training import SACActor, DEVICE

        env = MetaDriveEnv(config=env_config(config))
        actor = SACActor(config.obs_dim, config.action_dim, config.hidden_dim).to(DEVICE)
        reader = DanceReader(PheromoneTableV2(grid_size=config.grid_size),
                             num_lookahead_cells=config.lookahead_cells)
        shaping = ShapingCalculator(config)
        decay_rate = decay_rate_from_half_life(config.half_life_frac * config.phase1_total_steps)

        while True:
            # Apply the newest control messages. Stop on None.
            stop = False
            while True:
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


def run_multi_bee(config=default_config, num_bees=3, sync_every_updates=200):
    from single_bee_training import SACAgent, ReplayBuffer, write_success_path, write_crash_point

    decay_rate = decay_rate_from_half_life(config.half_life_frac * config.phase1_total_steps)
    budget = config.phase1_total_steps

    ctx = mp.get_context("spawn")
    counter = ctx.Value("l", 0)  # learner's transition count; workers read it for the reading gate
    result_q = ctx.Queue()
    control_qs = [ctx.Queue() for _ in range(num_bees)]
    workers = [ctx.Process(target=worker_main, args=(i, control_qs[i], result_q, counter, config),
                           daemon=True)
               for i in range(num_bees)]

    agent = SACAgent(config)
    buffer = ReplayBuffer(capacity=config.buffer_capacity)
    table = PheromoneTableV2(grid_size=config.grid_size)
    tracker = ReputationTracker(agent_ids=list(range(num_bees)))

    def broadcast_weights():
        state = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
        for q in control_qs:
            q.put(("weights", state))

    def broadcast_table():
        snapshot = copy.deepcopy(table)  # a fresh copy, so queue pickling never races a write
        for q in control_qs:
            q.put(("table", snapshot))

    print(f"\nMULTI-BEE // {num_bees} workers | budget {budget} steps | reward_scale={config.reward_scale}")
    for p in workers:
        p.start()
    broadcast_weights()
    broadcast_table()

    total_steps, updates, episodes, active = 0, 0, 0, num_bees
    try:
        while active > 0 and total_steps < budget:
            try:
                kind, worker_id, payload = result_q.get(timeout=1.0)
            except queue.Empty:
                if not any(p.is_alive() for p in workers):
                    break
                continue

            if kind == "error":
                print(f"  [Bee {worker_id}] ERROR\n{payload}")

            elif kind == "done":
                active -= 1

            elif kind == "transitions":
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
                        print(f"  [Update {updates} | step {total_steps}] critic={critic_loss:.4f} "
                              f"actor={actor_loss:.4f} entropy={entropy:.3f} alpha={alpha:.3f}")

            elif kind == "episode":
                episodes += 1
                arrived = payload["arrived"]
                crash_type = payload["crash_type"]
                path = payload["path"]
                caste = tracker.caste(worker_id)  # the caste this bee drove with
                vigor = tracker.vigor(worker_id)
                if arrived and caste != "NOVICE":
                    write_success_path(table, path, vigor, total_steps, decay_rate)
                elif crash_type is not None:
                    write_crash_point(table, path, crash_type, vigor, total_steps, decay_rate)
                tracker.update(worker_id, success=arrived)
                broadcast_table()

                path_m = sum(math.hypot(b[0][0] - a[0][0], b[0][1] - a[0][1])
                             for a, b in zip(path, path[1:]))
                terminal = "arrive_dest" if arrived else (crash_type or "timeout")
                print(f"  Ep {episodes:03d} | bee {worker_id} | caste={caste:<6} | terminal={terminal:<14} | "
                      f"steps={payload['steps']:3d} | path={path_m:6.2f} m | "
                      f"raw_R={payload['raw_return']:8.2f} | step={total_steps}")
    finally:
        for q in control_qs:
            q.put(None)
        for p in workers:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()

    print(f"\nMulti-bee run finished: {total_steps} transitions, {updates} updates, {episodes} episodes")
    print(f"Pheromone cells: {len(table.table)} | castes: {[tracker.caste(i) for i in range(num_bees)]}")
    return agent, table, tracker


if __name__ == "__main__":
    run_multi_bee(default_config, num_bees=3)
