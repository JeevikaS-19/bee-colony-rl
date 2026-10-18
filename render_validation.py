"""
render_validation.py

Loads the saved sandbox bee (sandbox_bee_checkpoint.pt) and its pheromone
table (pheromone_table.pkl), then runs a few episodes with rendering ON and
deterministic actions (the policy's best guess, no exploration noise).

What to look for:
  1. Does the car actually move? (total distance travelled is printed per episode)
  2. Do the printed trace readings line up with where the car is on screen?
  3. Do warning traces sit near spots where crashes actually happened?

Run from the same folder as sandbox_bee2_training.py and pheromone_table.py.
"""

import os
import sys
import time
import math
import pickle
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pheromone_table import PheromoneTable
from sandbox_bee2_training import SACAgent, DEVICE

try:
    from metadrive import MetaDriveEnv
    METADRIVE_AVAILABLE = True
except ImportError:
    METADRIVE_AVAILABLE = False

CHECKPOINT_PATH = "sandbox_bee_checkpoint.pt"
PHEROMONE_PATH = "pheromone_table.pkl"


def load_agent(obs_dim=259, action_dim=2):
    agent = SACAgent(state_dim=obs_dim, action_dim=action_dim)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.critic_target.load_state_dict(ckpt["critic_target"])
    agent.log_alpha.data.copy_(ckpt["log_alpha"].data)
    agent.actor.eval()
    return agent


def load_pheromone_table():
    table = PheromoneTable(grid_size=2.0)
    with open(PHEROMONE_PATH, "rb") as f:
        table.table = pickle.load(f)
    return table


def run_render_validation(num_episodes=5, step_delay=0.02, print_every=10):
    if not METADRIVE_AVAILABLE:
        print("CRITICAL: MetaDrive is not installed in this environment.")
        return

    for path in (CHECKPOINT_PATH, PHEROMONE_PATH):
        if not os.path.exists(path):
            print(f"Missing file: {path} -- run the training script first.")
            return

    agent = load_agent()
    table = load_pheromone_table()

    print("=== Loaded pheromone table ===")
    table.print_summary()
    print()

    config = {
        "map": "C",
        "use_render": True,
        "horizon": 500,  # keep in sync with sandbox_bee2_training.py
        "out_of_road_penalty": 20.0,
        "crash_vehicle_penalty": 20.0,
        "crash_object_penalty": 20.0,
    }

    env = MetaDriveEnv(config=config)
    try:
        for ep in range(1, num_episodes + 1):
            obs = env.reset()
            if isinstance(obs, tuple) and len(obs) == 2:
                obs, info = obs

            start_pos = tuple(env.agent.position)
            prev_pos = start_pos
            distance = 0.0
            ep_reward = 0.0
            print(f"--- Episode {ep} | spawn at ({start_pos[0]:.2f}, {start_pos[1]:.2f}) ---")

            for step in range(1, config["horizon"] + 1):
                action, _ = agent.actor.sample_action(obs, deterministic=True)
                step_results = env.step(action)
                if len(step_results) == 5:
                    obs, reward, terminated, truncated, info = step_results
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_results

                ep_reward += reward
                pos = tuple(env.agent.position)
                distance += math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1])
                prev_pos = pos

                if step % print_every == 0:
                    sensed = table.read_nearby(pos, decay_rate=0.0)
                    speed = getattr(env.agent, "speed_km_h", float("nan"))
                    print(f"  step {step:03d} | pos=({pos[0]:.2f}, {pos[1]:.2f}) | "
                          f"speed={speed:.1f} km/h | action=({action[0]:+.2f}, {action[1]:+.2f}) | "
                          f"sensed success={sensed['success']:.2f} warning={sensed['warning']:.2f}")

                time.sleep(step_delay)
                if done:
                    break

            reason = ("arrive_dest" if info.get("arrive_dest", False) else
                      "crash" if (info.get("crash_vehicle", False)
                                  or info.get("crash_object", False)
                                  or info.get("crash_building", False)
                                  or info.get("out_of_road", False))
                      else "timeout")
            print(f"  Episode {ep} done at step {step} | reason={reason} | "
                  f"reward={ep_reward:.3f} | total distance travelled={distance:.2f} m\n")
    finally:
        env.close()


if __name__ == "__main__":
    run_render_validation()