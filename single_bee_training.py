#phase 1-3
import sys
import os
import math
import time
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from config import BeeConfig, default_config
from pheromone_table_v2 import PheromoneTableV2, decay_rate_from_half_life
from dance_reader import DanceReader
from shaping_calculator import ShapingCalculator
from instrumentation import BeeInstrumentation

METADRIVE_AVAILABLE = False
try:
    from metadrive import MetaDriveEnv
    METADRIVE_AVAILABLE = True
except ImportError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        log_std = torch.clamp(self.log_std_linear(x), min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample_action(self, obs, deterministic=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(DEVICE)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        mean, log_std = self.forward(obs)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action.detach().cpu().numpy()[0], log_prob

    def sample(self, obs):
        """Batched training path for update_parameters(). Full-batch tensors, gradients kept."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_raw = normal.rsample()
        action = torch.tanh(action_raw)
        log_prob = normal.log_prob(action_raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, obs_dim=259, action_dim=2, hidden_dim=256):
        super(SACCritic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        # Q2
        self.fc3 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        xu = torch.cat([obs, action], dim=1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1_out(x1)

        x2 = F.relu(self.fc3(xu))
        x2 = F.relu(self.fc4(x2))
        q2 = self.q2_out(x2)
        return q1, q2


class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, config: BeeConfig = default_config):
        self.config = config
        self.actor = SACActor(config.obs_dim, config.action_dim, config.hidden_dim).to(DEVICE)
        self.critic = SACCritic(config.obs_dim, config.action_dim, config.hidden_dim).to(DEVICE)
        self.critic_target = SACCritic(config.obs_dim, config.action_dim, config.hidden_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.target_entropy = config.target_entropy
        self.log_alpha = torch.tensor(math.log(config.alpha_init), requires_grad=True, device=DEVICE)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.alpha_floor = config.alpha_floor

    def update_parameters(self, buffer, batch_size=256):
        if len(buffer) < batch_size:
            return None, None, None, None, self.log_alpha.exp().item()

        state, action, reward, next_state, done = buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_target, q2_target = self.critic_target(next_state, next_action)
            q_target = torch.min(q1_target, q2_target) - self.log_alpha.exp() * next_log_prob
            q_target = reward + (1.0 - done) * self.config.gamma * q_target

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        min_log_alpha = math.log(self.alpha_floor)
        self.log_alpha.data.clamp_(min=min_log_alpha)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau)

        return (critic_loss.item(), actor_loss.item(), alpha_loss.item(),
                -log_prob.detach().mean().item(), self.log_alpha.exp().item())

# MetaDrive crash flags. There is no generic "crash" key.
CRASH_FLAGS = ["out_of_road", "crash_vehicle", "crash_object", "crash_building"]
# Only these end an episode for real (done=1 in the buffer). Horizon timeouts bootstrap.
REAL_TERMINAL_FLAGS = ["arrive_dest"] + CRASH_FLAGS

def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0

def stdev(xs):
    m = avg(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else 0.0

def is_real_terminal(info):
    return any(info.get(flag, False) for flag in REAL_TERMINAL_FLAGS)

def reset_env(env):
    result = env.reset()
    return result if isinstance(result, tuple) else (result, {})

def write_success_path(table, episode_path, vigor, step, decay_rate):
    """Arrival: a success dance at every point on the path, progress rising to 1.0."""
    if not episode_path:
        return
    n = len(episode_path)
    for i, (pos, heading, steer, speed) in enumerate(episode_path):
        table.write_dance(pos, heading=heading, channel="success", steer=steer, vigor=vigor,
                          step=step, decay_rate=decay_rate, speed=speed, progress_after=(i + 1) / n)

def write_crash_point(table, episode_path, crash_type, vigor, step, decay_rate):
    """Crash: one dance at the last point. v2 routes the MetaDrive key to its crash channel."""
    if not episode_path:
        return
    pos, heading, steer, speed = episode_path[-1]
    table.write_dance(pos, heading=heading, channel="crash", steer=steer, vigor=vigor,
                      step=step, decay_rate=decay_rate, speed=speed, progress_after=1.0,
                      crash_type=crash_type)

def run_single_bee_pipeline(config: BeeConfig=default_config, max_episodes=None):
    print("\n"+"="*65)
    print("PROJECT BEE // SINGLE BEE PHASE 1-3 PIPELINE (v3)")
    print("="*65 + "\n")

    table = PheromoneTableV2(grid_size=config.grid_size)
    reader = DanceReader(table, num_lookahead_cells=config.lookahead_cells)
    shaping = ShapingCalculator(config)
    instrumentation = BeeInstrumentation(log_dir=config.get_phase_dir("single_bee"))

    agent = SACAgent(config)
    buffer = ReplayBuffer(capacity=config.buffer_capacity)

    print(f"Initialized Agent, PheromoneTableV2 (grid={config.grid_size}m), "
          f"DanceReader ({config.lookahead_cells} cells), and ShapingCalculator.")
    print("Initial Reading State: DISABLED (Phase 1-2 Write-Only)\n")

    if not METADRIVE_AVAILABLE:
        print("MetaDrive not available in current environment — pipeline scaffolding verified.")
        return table, reader, shaping, agent, buffer

    shaping.set_reading_enabled(False)

    total_steps = 0
    episodes_done = 0
    decay_rate = decay_rate_from_half_life(config.half_life_frac * config.phase1_total_steps)

    def read_state(env, step):
        """Position, heading, speed, and the corridor read at the env's current state.
        Reading and shaping stay off until config.read_start_step."""
        reading_enabled = step >= config.read_start_step
        shaping.set_reading_enabled(reading_enabled)
        pos = tuple(env.agent.position)
        heading = env.agent.heading_theta
        speed = env.agent.speed_km_h
        readout = reader.read_corridor(position=pos, heading_angle=heading,
                                       step_now=step, decay_rate=decay_rate,
                                       reading_enabled=reading_enabled)
        return pos, heading, speed, readout

    def potential_now(env, step):
        _, heading, speed, readout = read_state(env, step)
        return shaping.compute_potential(heading, speed, readout)

    print("STARTING PHASE 1")

    env = MetaDriveEnv(config={
        "map": config.map_type,
        "use_render": config.use_render,
        "horizon": config.horizon,
        # Same penalties as sandbox_bee2_training.py. MetaDrive's 5.0 default allowed reward hacking.
        "out_of_road_penalty": config.penalty,
        "crash_vehicle_penalty": config.penalty,
        "crash_object_penalty": config.penalty,
    })
    obs, info = reset_env(env)

    episode_reward = 0.0
    episode_path = []
    ep_throttle, ep_speed, ep_shaped, ep_det = [], [], [], []

    # Warmup: throttle-biased random actions seed the buffer with forward motion.
    # Transitions go into the buffer only; no pheromone writes from warmup.
    print(f"Warmup: {config.warmup_steps} steps of throttle-biased random exploration...")
    for _ in range(config.warmup_steps):
        action = np.array([np.random.uniform(-1, 1), np.random.uniform(0.3, 1.0)])
        step_res = env.step(action)
        if len(step_res) == 5:
            next_obs, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_res
        buffer.push(obs, action, config.reward_scale * reward, next_obs, float(is_real_terminal(info)))
        total_steps += 1
        if done:
            obs, info = reset_env(env)
        else:
            obs = next_obs
    obs, info = reset_env(env)
    phi_s = potential_now(env, total_steps)
    print("Warmup complete. Starting SAC training...")

    while total_steps < config.phase1_total_steps and (max_episodes is None or episodes_done < max_episodes):
        total_steps += 1
        if total_steps == config.read_start_step:
            print(f"\n -> STEP {total_steps}: READING AND SHAPING ENABLED")
        t_start = instrumentation.start_timer()
        action, _ = agent.actor.sample_action(obs)
        steering, throttle = action[0], action[1]
        det_action, _ = agent.actor.sample_action(obs, deterministic=True)
        det_throttle = det_action[1]

        step_res = env.step(action)
        if len(step_res) == 5:
            next_obs, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else: 
            next_obs, reward, done, info = step_res
        is_terminal = is_real_terminal(info)
        pos, heading, speed, readout = read_state(env, total_steps)
        episode_path.append((pos, heading, steering, speed))

        # Potential at the state after the step. Zero at a real terminal.
        phi_next = 0.0 if is_terminal else shaping.compute_potential(heading, speed, readout)
        shaped_reward = config.reward_scale * shaping.compute_shaped_reward(reward, phi_s, phi_next, gamma=config.gamma)

        buffer.push(obs, action, shaped_reward, next_obs, float(is_terminal))
        episode_reward += reward  # raw env reward for the episode log
        ep_throttle.append(throttle)
        ep_speed.append(speed)
        ep_shaped.append(shaped_reward)
        ep_det.append(det_throttle)
        phi_s = phi_next
        obs = next_obs

        instrumentation.record_step(
            start_time=t_start,
            probed_cells=len(readout['corridor_cells']),
            hit_cells=len(readout['raw_success_dances']) + len(readout['raw_crash_dances']),
            chosen_steer=steering,
            consensus_steer_log=readout['consensus_steer_log']   
        )

        if len(buffer) >= config.warmup_steps:
            critic_loss, actor_loss, alpha_loss, entropy, alpha_val = agent.update_parameters(buffer, config.batch_size)
            if critic_loss is not None and total_steps % 500 == 0:
                rewards = np.array([t[2] for t in buffer.buffer])
                actions = np.stack([t[1] for t in buffer.buffer])
                print(f"  [Step {total_steps}] critic={critic_loss:.4f} actor={actor_loss:.4f} "
                      f"alpha_loss={alpha_loss:+.4f} entropy={entropy:.3f} alpha={alpha_val:.3f} | "
                      f"buffer reward mean={rewards.mean():+.4f} std={rewards.std():.4f} | "
                      f"steer mean={actions[:, 0].mean():+.3f} std={actions[:, 0].std():.3f} | "
                      f"throttle mean={actions[:, 1].mean():+.3f} std={actions[:, 1].std():.3f}")

        if done:
            arrived = info.get("arrive_dest", False)
            crashed = any(info.get(flag, False) for flag in CRASH_FLAGS)

            #Phase 2: Write dances to PheromoneTableV2
            vigor = 1.0  
            if arrived:
                write_success_path(table, episode_path, vigor, total_steps, decay_rate)
            elif crashed:
                crash_type = next(flag for flag in CRASH_FLAGS if info.get(flag, False))
                write_crash_point(table, episode_path, crash_type, vigor, total_steps, decay_rate)

            instrumentation.record_episode_end(episode_reward, arrived, len(episode_path))

            episodes_done += 1
            path_m = sum(math.hypot(b[0][0] - a[0][0], b[0][1] - a[0][1])
                         for a, b in zip(episode_path, episode_path[1:]))
            terminal = "arrive_dest" if arrived else next((f for f in CRASH_FLAGS if info.get(f, False)), "timeout")
            print(f"  Episode {episodes_done:03d} | steps={len(episode_path):3d} | terminal={terminal:<14} "
                  f"| path={path_m:7.2f} m | step={total_steps} | speed={avg(ep_speed):5.2f} km/h "
                  f"| thr_mean={avg(ep_throttle):+.3f} thr_std={stdev(ep_throttle):.3f} thr_det={avg(ep_det):+.3f} "
                  f"| alpha={agent.log_alpha.exp().item():.3f} | raw_R={episode_reward:8.2f} | shaped_R={sum(ep_shaped):8.2f}")
            ep_throttle, ep_speed, ep_shaped, ep_det = [], [], [], []


            episode_reward = 0.0
            episode_path = []
            obs, info = reset_env(env)
            phi_s = potential_now(env, total_steps)

    env.close()
    instrumentation.save_summary_json("phase1_3_summary.json")
    print(f"\nPipeline completed at step {total_steps}. Final Dance Table entries: {len(table.table)}")
    return table, reader, shaping, agent, buffer


if __name__ == "__main__":
    table, reader, shaping, agent, buffer = run_single_bee_pipeline()
    print("Pipeline scaffolding verified successfully!")