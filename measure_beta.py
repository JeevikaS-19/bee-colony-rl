"""
measure_beta.py

Sets beta for the shaping arms by the written rule in Documentation/shaping_spec.md
section 10 (beta rule):

    beta such that mean |F| = TARGET * mean |r|,
    measured on a rollout of a trained policy with the trace table rescaled
    so every trace has vigor 1.0.

F is linear in beta, so one rollout at beta = 1 gives the answer:
    beta = TARGET * mean|r| / mean|F_at_beta_1|

TARGET = 0.10 is a judgment call (see the spec). Rescaling to vigor 1.0 removes
the caste weighting, so the scale does not depend on how often a bee was promoted.

Decay is off here: the table is read as a fixed final snapshot.
"""

import argparse
import math
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from metadrive import MetaDriveEnv
from pheromone_table import PheromoneTable
from sandbox_bee2_training import SACAgent, DEVICE, classify_terminal, is_real_terminal
from shaping import potential, shaping_term

TARGET = 0.10
GAMMA = 0.99


def load_agent(path):
    agent = SACAgent(state_dim=259, action_dim=2)
    ckpt = torch.load(path, map_location=DEVICE)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.actor.eval()
    return agent


def load_table_rescaled(path):
    table = PheromoneTable(grid_size=2.0)
    with open(path, "rb") as f:
        table.table = pickle.load(f)
    # Rescale every written trace to vigor 1.0 (keep the cell layout).
    for cell in table.table.values():
        for channel in ("success", "warning"):
            if cell[channel]["vigor"] > 0.0:
                cell[channel]["vigor"] = 1.0
    return table


def measure(checkpoint, table_path, episodes, seed):
    agent = load_agent(checkpoint)
    table = load_table_rescaled(table_path)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = {
        "map": "C",
        "use_render": False,
        "horizon": 500,
        "out_of_road_penalty": 20.0,
        "crash_vehicle_penalty": 20.0,
        "crash_object_penalty": 20.0,
    }

    abs_r, abs_F, r_all, phi_nonzero, steps_total = [], [], [], 0, 0
    env = MetaDriveEnv(config=config)
    try:
        for ep in range(1, episodes + 1):
            obs = env.reset()
            if isinstance(obs, tuple) and len(obs) == 2:
                obs, info = obs
            x, y = env.agent.position
            phi_cur = potential(*_read(table, (x, y)), beta=1.0, lam=1.0)
            ep_steps, ep_dist, prev = 0, 0.0, (x, y)
            info = {}
            while True:
                action, _ = agent.actor.sample_action(obs, deterministic=False)
                res = env.step(action)
                if len(res) == 5:
                    obs, reward, terminated, truncated, info = res
                    done = terminated or truncated
                else:
                    obs, reward, done, info = res
                x, y = env.agent.position
                ep_dist += math.hypot(x - prev[0], y - prev[1])
                prev = (x, y)
                phi_next = potential(*_read(table, (x, y)), beta=1.0, lam=1.0)
                F = shaping_term(phi_cur, phi_next, GAMMA, next_is_terminal=is_real_terminal(info))
                phi_cur = phi_next
                abs_F.append(abs(F))
                abs_r.append(abs(reward))
                r_all.append(reward)
                if phi_next != 0.0:
                    phi_nonzero += 1
                steps_total += 1
                ep_steps += 1
                if done or ep_steps >= 500:
                    break
            reason, _ = classify_terminal(info)
            print(f"  episode {ep}: steps={ep_steps} reason={reason} distance={ep_dist:.1f} m")
    finally:
        env.close()

    mean_abs_r = float(np.mean(abs_r))
    mean_abs_F = float(np.mean(abs_F))
    beta = TARGET * mean_abs_r / mean_abs_F if mean_abs_F > 0 else float("inf")
    print(f"\nsteps measured: {steps_total}")
    print(f"mean |r| per step (raw env reward): {mean_abs_r:.4f}")
    print(f"mean |F| per step at beta=1:        {mean_abs_F:.4f}")
    print(f"fraction of steps with Phi != 0:    {phi_nonzero / max(1, steps_total):.3f}")
    print(f"traces in table (vigor set to 1.0): {sum(1 for c in table.table.values() if c['success']['vigor'] > 0 or c['warning']['vigor'] > 0)} cells")
    print(f"\nRULE: beta = {TARGET} * mean|r| / mean|F_beta1| = {beta:.3f}")
    return beta


def _read(table, xy):
    r = table.read_nearby(xy, step=0, decay_rate=0.0)
    return r["success"], r["warning"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to sandbox_bee_checkpoint.pt (copy it first)")
    parser.add_argument("--table", required=True, help="path to pheromone_table.pkl (copy it first)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    measure(args.checkpoint, args.table, args.episodes, args.seed)
