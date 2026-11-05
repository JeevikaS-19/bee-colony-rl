"""
Frozen-policy evaluation. Loads the actor from a training checkpoint, runs a fixed number of
episodes on each requested scenario seed, with NO pheromone table, no shaping and no learning.

Per episode: route_completion (MetaDrive info['route_completion'], 0-1), path_length_m (sum of
step displacements), displacement_m (straight line start to end), arrival, terminal.
Terminal: arrive_dest | out_of_road | crash_vehicle | crash_object | crash_building | timeout.

Seeds listed in --train-seeds are labelled "train" (memorisation); the rest "heldout" (generalisation).
The episode order is a shuffle drawn from --seed, so two arms evaluated with the same --seed and
seed list see the same schedule (paired comparison). Actions are the deterministic mean unless
--stochastic is given.

Usage
  python eval_policy.py --checkpoint DIR/checkpoint_100k --eval-seeds 0,2,5,6,7 --train-seeds 0,1,2,3,4 \
      --episodes-per-scenario 20 --out-dir eval_results/NAME
  (DIR/checkpoint_100k must contain agent.pt)
"""
import argparse
import csv
import json
import math
import os

import numpy as np


def stats(rows):
    n = len(rows)
    if n == 0:
        return None
    c = lambda t: sum(1 for r in rows if r["terminal"] == t)
    crashes = sum(1 for r in rows if r["terminal"] not in ("arrive_dest", "timeout"))
    return {"episodes": n,
            "arrival_rate": round(c("arrive_dest") / n, 3),
            "crash_rate": round(crashes / n, 3),
            "timeout_rate": round(c("timeout") / n, 3),
            "avg_route_completion_pct": round(100 * sum(r["route_completion"] for r in rows) / n, 1),
            "avg_path_length_m": round(sum(r["path_length_m"] for r in rows) / n, 1),
            "avg_displacement_m": round(sum(r["displacement_m"] for r in rows) / n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="folder containing agent.pt")
    ap.add_argument("--eval-seeds", default="0,2,5,6,7")
    ap.add_argument("--train-seeds", default="0,1,2,3,4", help="seeds the policy was trained on (labelling only)")
    ap.add_argument("--episodes-per-scenario", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0, help="seeds the episode order and torch")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    args = ap.parse_args()

    import torch
    from metadrive import MetaDriveEnv
    from config import default_config as config
    from single_bee_training import SACActor, DEVICE, CRASH_FLAGS
    from multi_bee_training import env_config

    eval_seeds = [int(x) for x in args.eval_seeds.split(",")]
    train_seeds = {int(x) for x in args.train_seeds.split(",")} if args.train_seeds else set()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(os.path.join(args.checkpoint, "agent.pt"), map_location="cpu")
    actor = SACActor(config.obs_dim, config.action_dim, config.hidden_dim)
    actor.load_state_dict(ckpt["actor"])
    actor.to(DEVICE).eval()
    torch.manual_seed(args.seed)

    schedule = [s for s in eval_seeds for _ in range(args.episodes_per_scenario)]
    np.random.RandomState(args.seed).shuffle(schedule)

    env = MetaDriveEnv(config=env_config(config, 0, max(eval_seeds) + 1))
    rows = []
    try:
        for i, sd in enumerate(schedule, 1):
            res = env.reset(seed=sd)
            obs = res[0] if isinstance(res, tuple) else res
            p0 = tuple(env.agent.position)
            path_len, prev, done, info = 0.0, p0, False, {}
            while not done:
                with torch.no_grad():
                    action, _ = actor.sample_action(obs, deterministic=not args.stochastic)
                action = action.cpu().numpy().reshape(-1) if hasattr(action, "cpu") else np.asarray(action).reshape(-1)
                out = env.step(action)
                if len(out) == 5:
                    obs, _, term, trunc, info = out
                    done = term or trunc
                else:
                    obs, _, done, info = out
                pos = tuple(env.agent.position)
                path_len += math.hypot(pos[0] - prev[0], pos[1] - prev[1])
                prev = pos
            arrived = bool(info.get("arrive_dest", False))
            terminal = "arrive_dest" if arrived else next((f for f in CRASH_FLAGS if info.get(f, False)),
                                                           "out_of_road" if info.get("out_of_road", False) else "timeout")
            row = {"episode": i, "scenario_seed": sd, "split": "train" if sd in train_seeds else "heldout",
                   "route_completion": float(info.get("route_completion", 0.0)),
                   "path_length_m": round(path_len, 2),
                   "displacement_m": round(math.hypot(prev[0] - p0[0], prev[1] - p0[1]), 2),
                   "arrival": arrived, "terminal": terminal}
            rows.append(row)
            print(f"Episode {i}/{len(schedule)}: scenario seed = {sd} ({row['split']}) | {terminal:<13} | "
                  f"route_completion {100 * row['route_completion']:.0f}% | path {path_len:.0f} m", flush=True)
    finally:
        env.close()

    with open(os.path.join(args.out_dir, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"checkpoint": args.checkpoint, "checkpoint_total_steps": ckpt.get("total_steps"),
               "deterministic": not args.stochastic, "order_seed": args.seed,
               "per_scenario": {str(sd): {"split": "train" if sd in train_seeds else "heldout",
                                          **stats([r for r in rows if r["scenario_seed"] == sd])}
                                for sd in eval_seeds},
               "heldout": stats([r for r in rows if r["split"] == "heldout"]),
               "train_roads": stats([r for r in rows if r["split"] == "train"]),
               "overall": stats(rows)}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nPER SCENARIO")
    for k, v in summary["per_scenario"].items():
        print(f"  seed {k} [{v['split']}]: {v}")
    for k in ("train_roads", "heldout", "overall"):
        print(f"  {k}: {summary[k]}")


if __name__ == "__main__":
    main()
