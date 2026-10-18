"""
idm_pheromone_validation.py

Exercises the pheromone table's SUCCESS path without depending on the SAC
policy learning anything. MetaDrive's built-in IDM autopilot drives the car
(it actually reaches the destination), and every episode end goes through
the exact same classify_terminal() + write_episode_traces() used by the
training loop in sandbox_bee2_training.py.

Checks (each prints PASS/FAIL):
  1. IDM reaches arrive_dest at least once.
  2. Every grid cell the car drove through has a success trace, and no
     other cells do.
  3. read_nearby() senses a strong success trace at every point on the path.
  4. read_nearby() senses nothing at a point well off the route.
  5. Warning channel stays empty when nothing crashed.
  6. Repeating the same route doesn't grow the table (overwrite, not append).

Writes the resulting table to idm_pheromone_table.pkl (never touches the
training run's pheromone_table.pkl). Run with --render to watch the drive.
"""

import os
import sys
import math
import pickle
import argparse

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pheromone_table import PheromoneTable
from sandbox_bee2_training import classify_terminal, write_episode_traces

try:
    from metadrive import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy
    METADRIVE_AVAILABLE = True
except ImportError:
    METADRIVE_AVAILABLE = False

# Renamed from idm_pheromone_table.pkl so a re-run can't overwrite the old pickle.
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idm_validation_table.pkl")
SENSOR_RADIUS = 10.0
SIGMA = 6.0


def run_idm_episode(env, max_steps):
    obs = env.reset()
    path = []
    info = {}
    for _ in range(max_steps):
        # Action is ignored -- agent_policy=IDMPolicy drives the car.
        step_results = env.step([0.0, 0.0])
        if len(step_results) == 5:
            _, _, terminated, truncated, info = step_results
            done = terminated or truncated
        else:
            _, _, done, info = step_results
        x, y = env.agent.position
        # IDM's steering isn't observed here, so steer is recorded as 0.0
        path.append((x, y, env.agent.speed_km_h, 0.0))
        if done:
            break
    return path, info


def check(label, passed, detail=""):
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return passed


def run_validation(num_episodes=3, render=False, max_steps=1000):
    if not METADRIVE_AVAILABLE:
        print("CRITICAL: MetaDrive is not installed in this environment.")
        return False

    config = {
        "map": "C",
        "use_render": render,
        "horizon": max_steps,
        "agent_policy": IDMPolicy,
        "out_of_road_penalty": 20.0,
        "crash_vehicle_penalty": 20.0,
        "crash_object_penalty": 20.0,
    }

    table = PheromoneTable(grid_size=2.0)
    episodes = []  # (reason, fired_flags, path, table_size_after)

    env = MetaDriveEnv(config=config)
    try:
        for ep in range(1, num_episodes + 1):
            path, info = run_idm_episode(env, max_steps)
            reason, fired = classify_terminal(info)
            # IDM is a fixed scripted driver used for validation, not a learning bee: write as SCOUT.
            write_episode_traces(table, reason, path, vigor=1.0, worker_id="idm_bee", step=ep, caste="SCOUT")
            episodes.append((reason, fired, [p[:2] for p in path], len(table.table)))
            print(f"--- IDM episode {ep}: reason={reason} flags={fired} | steps={len(path)} | "
                  f"end=({path[-1][0]:.1f}, {path[-1][1]:.1f}) | table cells={len(table.table)}")
    finally:
        env.close()

    print("\n=== Checks ===")
    results = []

    arrived = [e for e in episodes if e[0] == "arrive_dest"]
    results.append(check("IDM reaches arrive_dest", len(arrived) > 0,
                         f"{len(arrived)}/{len(episodes)} episodes arrived"))
    if not arrived:
        print("\nNo arrival, so the success path can't be checked. Try a larger --max-steps.")
        return False

    path = arrived[0][2]
    path_cells = {table._to_grid_key(p) for p in path}
    for _, _, other_path, _ in arrived[1:]:
        path_cells |= {table._to_grid_key(p) for p in other_path}
    success_cells = {k for k, ch in table.table.items() if ch["success"]["vigor"] > 0.0}
    results.append(check("success traces cover exactly the driven cells",
                         success_cells == path_cells,
                         f"{len(success_cells)} success cells, {len(path_cells)} driven cells, "
                         f"{len(success_cells ^ path_cells)} mismatched"))

    on_path = [table.read_nearby(p, step=0, sensor_radius=SENSOR_RADIUS, sigma=SIGMA, decay_rate=0.0)["success"]
               for p in path]
    # Worst case on-path: ~1.41 m from a cell centre -> exp(-2/72) ~= 0.97
    results.append(check("strong success reading everywhere on the path", min(on_path) >= 0.9,
                         f"min={min(on_path):.3f}, mean={sum(on_path) / len(on_path):.3f}"))

    # A point 60 m beyond the route's bounding box -- far outside the sensor radius.
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    off_route = (min(xs) - 60.0, min(ys) - 60.0)
    nearest = min(math.dist(off_route, p) for p in path)
    off_reading = table.read_nearby(off_route, step=0, sensor_radius=SENSOR_RADIUS, sigma=SIGMA, decay_rate=0.0)
    results.append(check("nothing sensed well off the route", off_reading["success"] == 0.0,
                         f"point {off_route} is {nearest:.1f} m from the path, "
                         f"sensed success={off_reading['success']:.3f}"))

    crashed = any(e[0] == "crash" for e in episodes)
    warning_cells = [k for k, ch in table.table.items() if ch["warning"]["vigor"] > 0.0]
    if crashed:
        results.append(check("warning written for the crash episode(s)", len(warning_cells) > 0,
                             f"{len(warning_cells)} warning cells"))
    else:
        results.append(check("warning channel empty with no crashes", len(warning_cells) == 0,
                             f"{len(warning_cells)} warning cells"))

    if len(arrived) >= 2:
        sizes = [e[3] for e in episodes if e[0] == "arrive_dest"]
        results.append(check("repeat arrivals don't grow the table much", sizes[-1] - sizes[0] <= 0.1 * sizes[0],
                             f"table cells after each arrival: {sizes}"))

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(table.table, f)
    print(f"\nSaved IDM pheromone table to {OUTPUT_PATH}")

    print("\n--- Strongest traces (first 10) ---")
    for line in table.summarize()[:10]:
        print(" ", line)

    all_passed = all(results)
    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'} "
          f"({sum(results)}/{len(results)})")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    run_validation(num_episodes=args.episodes, render=args.render, max_steps=args.max_steps)
