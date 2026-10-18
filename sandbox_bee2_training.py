import sys
import os
import math
import random
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from pheromone_table import PheromoneTable, decay_rate_from_half_life

METADRIVE_AVAILABLE = False
try:
    from metadrive import MetaDriveEnv
    METADRIVE_AVAILABLE = True
except ImportError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CPU-only box: torch's default (10 threads here) oversubscribes the CPU and
# made each SAC update ~300 ms. Measured: 1 thread ~36 ms, 2 threads ~22 ms,
# 4+ threads 300-400 ms. Keep this small -- MetaDrive needs CPU too.
torch.set_num_threads(2)


class SACActor(nn.Module):
    def __init__(self, obs_dim=259, action_dim=2, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(SACActor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample_action(self, obs, deterministic=False):
        """
        Single-observation inference path -- used for live env stepping.
        Returns a detached numpy action, safe to hand straight to env.step().
        NOT used during training updates (see `sample()` below).
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(DEVICE)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        if deterministic:
            action_raw = mean
            action = torch.tanh(action_raw)
            log_prob = None
        else:
            normal = Normal(mean, std)
            action_raw = normal.rsample()
            action = torch.tanh(action_raw)
            log_prob = normal.log_prob(action_raw) - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action.detach().cpu().numpy()[0], log_prob

    def sample(self, obs):
        """
        Batched training path -- used inside SACAgent.update_parameters().
        Keeps everything as full-batch tensors WITH gradients intact
        (no .detach(), no .cpu()/.numpy(), no indexing down to one row).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_raw = normal.rsample()
        action = torch.tanh(action_raw)
        log_prob = normal.log_prob(action_raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, state_dim=259, action_dim=2):
        super(SACCritic, self).__init__()

        self.fc1_1 = nn.Linear(state_dim + action_dim, 256)
        self.fc1_2 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, 1)

        self.fc2_1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_2 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

    def forward(self, state, action):
        sa_input = torch.cat([state, action], dim=-1)

        x1 = F.relu(self.fc1_1(sa_input))
        x1 = F.relu(self.fc1_2(x1))
        q1 = self.q1_out(x1)

        x2 = F.relu(self.fc2_1(sa_input))
        x2 = F.relu(self.fc2_2(x2))
        q2 = self.q2_out(x2)

        return q1, q2


class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device="cpu"):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    # target_entropy: the policy entropy alpha steers toward. Alpha only falls
    # while entropy is ABOVE the target, so a target that's too low lets alpha
    # sink to the floor while the policy narrows. The textbook -action_dim
    # (-2.0) let alpha floor by ~13k steps and the bee settled into creeping.
    # 0.0 keeps roughly ~0.25 pre-tanh std per action dim -- enough throttle
    # noise to keep stumbling into "driving faster pays".
    def __init__(self, state_dim=259, action_dim=2, lr=3e-4, gamma=0.99, tau=0.005, target_entropy=0.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau

        self.actor = SACActor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_target = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = target_entropy
        self.log_alpha_min = math.log(0.03)  # floor: alpha can't collapse below 0.03
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_parameters(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None, None, None, None

        state, action, reward, next_state, done = replay_buffer.sample(batch_size, self.device)

        # --- Critic loss ---
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q_value = reward + (1.0 - done) * self.gamma * min_target_q

        current_q1, current_q2 = self.critic(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q_value)
        critic2_loss = F.mse_loss(current_q2, target_q_value)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor loss ---
        pi_action, log_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Entropy temperature (alpha) loss ---
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Floor alpha AFTER this call's own update, not before --
        # keeps the clamp from lagging one step behind.
        self.log_alpha.data.clamp_(min=self.log_alpha_min)

        # --- Soft update (Polyak) of target critic ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # Entropy estimate of the current policy (-mean log_pi). Compare it to
        # target_entropy: alpha falls while entropy > target, rises while below.
        entropy = -log_pi.detach().mean().item()

        return critic_loss.item(), actor_loss.item(), self.alpha.item(), entropy


TERMINAL_FLAGS = ["arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash_building", "max_step"]


def classify_terminal(info):
    """
    Turns MetaDrive's info dict at episode end into (reason, fired_flags).
    reason is one of "arrive_dest" / "crash" / "timeout"; fired_flags lists
    every terminal flag that was True, so the log shows WHICH kind of crash.

    There is no generic "crash" key in MetaDrive -- collisions are split into
    crash_vehicle / crash_object / crash_building, and leaving the road is
    out_of_road. Checking a "crash" key silently never fires.
    """
    fired = [flag for flag in TERMINAL_FLAGS if info.get(flag, False)]
    if "arrive_dest" in fired:
        reason = "arrive_dest"
    elif any(flag in fired for flag in ("out_of_road", "crash_vehicle", "crash_object", "crash_building")):
        reason = "crash"
    else:
        reason = "timeout"
    return reason, fired



# Flags that end an episode for real. Only these mark a transition as terminal
# for the critic (no bootstrapping past them). A horizon timeout (max_step) is a
# truncation, not a terminal: the next state still has value, so it must NOT be
# in this list.
REAL_TERMINAL_FLAGS = ["arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash_building"]


def is_real_terminal(info):
    return any(info.get(flag, False) for flag in REAL_TERMINAL_FLAGS)


def write_episode_traces(pheromone_table, reason, episode_path, vigor, worker_id, step):
    """
    Episode-end write hook. arrive_dest -> 'success' along the whole path;
    crash -> one 'warning' at the last position. Returns cells written.
    Shared by the training loop and idm_pheromone_validation.py, so the
    validation exercises this exact code rather than a copy of it.
    """
    if vigor <= 0:
        return 0  # muted (NOVICE) bees listen but never write
    if reason == "arrive_dest":
        written_cells = 0
        meta = {"worker_id": worker_id, "trigger": "arrive_dest", "path_length": len(episode_path)}
        for pos in episode_path:
            if pheromone_table.write(pos, "success", vigor=vigor, step=step, meta=meta):
                written_cells += 1
        print(f"[Write Hook] ARRIVED! Backtraced path and updated {written_cells} grid cells with 'success'.")
        return written_cells

    if reason == "crash":
        crash_pos = episode_path[-1] if episode_path else (0.0, 0.0)
        meta = {"worker_id": worker_id, "trigger": "crash", "episode_steps": len(episode_path)}
        committed = pheromone_table.write(crash_pos, "warning", vigor=vigor, step=step, meta=meta)
        print(f"    [Write Hook]  CRASH at {crash_pos}! Warning trace committed: {committed}")
        return 1 if committed else 0

    return 0


def warmup_action():
    """
    Throttle-biased random action for the warmup phase. A fresh actor's
    throttle is centred on 0 (half accelerate, half brake), so the replay
    buffer used to hold almost nothing but creeping. Biasing throttle to
    [0.2, 1.0] fills it with real speed -- and the crashes that come with
    it -- so the critic sees what driving is worth. Steering stays small and
    zero-mean so the car mostly holds its lane until the curve.
    """
    steering = random.uniform(-0.3, 0.3)
    throttle = random.uniform(0.2, 1.0)
    return np.array([steering, throttle], dtype=np.float32)


HERE = os.path.dirname(os.path.abspath(__file__))  # outputs go next to this script, not the cwd


def save_checkpoint(agent, pheromone_table, out_dir):
    ckpt_path = os.path.join(out_dir, "sandbox_bee_checkpoint.pt")
    table_path = os.path.join(out_dir, "pheromone_table.pkl")
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "log_alpha": agent.log_alpha,
    }, ckpt_path)
    with open(table_path, "wb") as f:
        pickle.dump(pheromone_table.table, f)
    print(f"[Checkpoint] Saved agent weights to {ckpt_path} and pheromone table to {table_path}")


def set_seeds(seed):
    # MetaDrive itself is already deterministic here (map C, num_scenarios=1);
    # these cover the network init, SAC sampling and warmup actions.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_sac_training(run_name, total_steps=20000, batch_size=256, warmup_steps=5000, update_every=1,
                     seed=0, save_every=10000, half_life_frac=0.15):
    """
    run_name: output folder under checkpoints/. Refuses to reuse a non-empty
      folder, so a run can never overwrite another run's weights.
    half_life_frac: pheromone half-life as a fraction of total_steps (0 = no decay).
      The decay is in training steps, counted on this process's env only. With
      several workers this must become one shared global env-step counter.
    """
    """
    Real single-worker SAC training loop for the sandbox bee.
    Pheromone table write-side wired in (vigor stubbed at 1.0 --
    single bee, no hierarchy needed yet).

    The first `warmup_steps` steps use throttle-biased random actions
    (see warmup_action) and no updates; SAC updates start once warmup ends.
    """
    print("INITIALIZE SANDBOX BEE // TRAINING MODE")

    if not METADRIVE_AVAILABLE:
        print("CRITICAL: MetaDrive is not installed in this environment.")
        return

    out_dir = os.path.join(HERE, "checkpoints", run_name)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"{out_dir} already has files. Pick a new --run-name.")
    os.makedirs(out_dir, exist_ok=True)

    set_seeds(seed)

    half_life_steps = half_life_frac * total_steps if half_life_frac > 0 else None
    decay_rate = decay_rate_from_half_life(half_life_steps)
    print(f"Run '{run_name}' -> {out_dir} | pheromone half-life: "
          f"{'none' if half_life_steps is None else f'{half_life_steps:.0f} steps'} "
          f"(decay_rate={decay_rate:.6f}/step; used once reading is wired in)")

    obs_dim = 259
    action_dim = 2

    agent = SACAgent(state_dim=obs_dim, action_dim=action_dim)
    buffer = ReplayBuffer(capacity=200000)

    pheromone_table = PheromoneTable(grid_size=2.0)

    config = {
        "map": "C",
        "use_render": False,
        "horizon": 500,  # was 200 -- IDM needs 293 steps to arrive on map C; at the 51k policy's ~0.5 m/step the route needs ~500
        # 5.0 (default) allowed reward hacking; 40.0 overcorrected into a
        # frozen policy (standing still was safest). Trying a middle value.
        "out_of_road_penalty": 20.0,
        "crash_vehicle_penalty": 20.0,
        "crash_object_penalty": 20.0,
    }

    # Record exactly what this run used, so an ablation can be checked later.
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"run_name": run_name, "total_steps": total_steps, "batch_size": batch_size,
                   "warmup_steps": warmup_steps, "seed": seed, "save_every": save_every,
                   "half_life_frac": half_life_frac, "decay_rate_per_step": decay_rate,
                   "env": config}, f, indent=2)

    env = MetaDriveEnv(config=config)
    try:
        obs = env.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, info = obs

        episode_reward = 0.0
        episode_count = 0
        recent_episode_rewards = deque(maxlen=20)

        episode_path = []

        # Distance travelled per episode -- lets a frozen policy show up in the
        # training log instead of only in a rendered eval.
        episode_distance = 0.0
        recent_episode_distances = deque(maxlen=20)
        prev_pos = tuple(env.agent.position)
        episode_max_speed = 0.0

        outcome_counts = {}  # e.g. {"out_of_road": 12, "max_step": 40, ...} over the whole run

        print(f"Initial observation shape: {np.asarray(obs).shape}")
        print(f"Training for {total_steps} steps | batch_size={batch_size} | seed={seed} | "
              f"target_entropy={agent.target_entropy} | "
              f"warmup={warmup_steps} throttle-biased random steps before updates begin\n")

        for step in range(1, total_steps + 1):
            in_warmup = step <= warmup_steps
            if in_warmup:
                action = warmup_action()
            else:
                action, _ = agent.actor.sample_action(obs, deterministic=False)

            steering, throttle = action[0], action[1]
            assert -1.0 <= steering <= 1.0, f"Steering out of bounds: {steering}"
            assert -1.0 <= throttle <= 1.0, f"Throttle out of bounds: {throttle}"

            step_results = env.step(action)
            if len(step_results) == 5:
                next_obs, reward, terminated, truncated, info = step_results
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_results

            current_x, current_y = env.agent.position
            episode_path.append((current_x, current_y))
            episode_distance += math.hypot(current_x - prev_pos[0], current_y - prev_pos[1])
            prev_pos = (current_x, current_y)
            episode_max_speed = max(episode_max_speed, env.agent.speed_km_h)

            # Only real terminals stop bootstrapping. A horizon timeout still
            # episode-ends below (for reset + traces) but its next state has value.
            buffer.push(obs, action, reward, next_obs, float(is_real_terminal(info)))
            episode_reward += reward
            obs = next_obs

            if not in_warmup and step % update_every == 0:
                critic_loss, actor_loss, alpha_val, entropy = agent.update_parameters(buffer, batch_size)
                if critic_loss is not None and step % 500 == 0:
                    print(f"[Step {step:06d}] critic_loss={critic_loss:.4f} "
                          f"actor_loss={actor_loss:.4f} alpha={alpha_val:.4f} "
                          f"entropy={entropy:.3f} (target {agent.target_entropy})")

            if done:
                episode_count += 1
                recent_episode_rewards.append(episode_reward)
                avg_recent = sum(recent_episode_rewards) / len(recent_episode_rewards)
                recent_episode_distances.append(episode_distance)
                avg_recent_distance = sum(recent_episode_distances) / len(recent_episode_distances)

                reason, fired = classify_terminal(info)
                for flag in (fired or ["none"]):
                    outcome_counts[flag] = outcome_counts.get(flag, 0) + 1

                phase = "warmup" if in_warmup else "policy"
                print(f"  Episode {episode_count:04d} finished at step {step:06d} [{phase}] | "
                      f"reason={reason} flags={fired} | episode_reward={episode_reward:.4f} | "
                      f"avg_last_20={avg_recent:.4f} | distance={episode_distance:.2f} m | "
                      f"avg_dist_last_20={avg_recent_distance:.2f} m | max_speed={episode_max_speed:.1f} km/h")

                vigor_stub = 1.0  # fixed value -- swap for tracker.vigor(worker_id) once multi-bee
                write_episode_traces(pheromone_table, reason, episode_path, vigor_stub, "sandbox_bee", step)

                episode_path = []
                episode_reward = 0.0
                episode_distance = 0.0
                episode_max_speed = 0.0
                obs = env.reset()
                if isinstance(obs, tuple) and len(obs) == 2:
                    obs, info = obs
                prev_pos = tuple(env.agent.position)  # new spawn point, not last episode's end

            # Periodic save, so a long run that dies partway still leaves something to load.
            if save_every and step % save_every == 0 and step != total_steps:
                save_checkpoint(agent, pheromone_table, out_dir)
    finally:
        env.close()
    print("\nSUCCESS: training loop completed.")
    print(f"Total episodes: {episode_count}")
    print(f"Terminal flag counts over the run: {outcome_counts}")
    terminal_transitions = sum(1 for t in buffer.buffer if t[4] == 1.0)
    print(f"Replay buffer: {len(buffer)} transitions, {terminal_transitions} marked terminal")

    # --- Pheromone table summary (plain-English readout) ---
    print("\n--- Pheromone Table Summary ---")
    pheromone_table.print_summary()

    # --- Persist the trained agent and the pheromone table ---
    save_checkpoint(agent, pheromone_table, out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--run-name", required=True, help="output folder under checkpoints/")
    parser.add_argument("--half-life-frac", type=float, default=0.15,
                        help="pheromone half-life as a fraction of --steps; 0 = no decay")
    args = parser.parse_args()
    run_sac_training(run_name=args.run_name, total_steps=args.steps, seed=args.seed,
                     warmup_steps=args.warmup, half_life_frac=args.half_life_frac)
