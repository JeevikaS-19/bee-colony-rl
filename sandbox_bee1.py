import sys
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
 
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
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std
 
    def sample_action(self, obs, deterministic=False):
        """
        Single-observation inference path — used for live env stepping.
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
        Batched training path — used inside SACAgent.update_parameters().
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
    def __init__(self, state_dim=259, action_dim=2, lr=3e-4, gamma=0.99, tau=0.005, target_entropy=-2.0):
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
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
 
    @property
    def alpha(self):
        return self.log_alpha.exp()
 
    def update_parameters(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None, None, None
 
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
 
        # --- Soft update (Polyak) of target critic ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
 
        return critic_loss.item(), actor_loss.item(), self.alpha.item()
 
 
def run_sac_training(total_steps=20000, batch_size=256, start_training_after=1000, update_every=1):
    """
    Real single-worker SAC training loop for the sandbox bee.
    No pheromones, no multiprocessing yet — just: does this bee learn
    from its own plain reward, on this environment, at all?
    """
    print("INITIALIZE SANDBOX BEE // TRAINING MODE")
 
    if not METADRIVE_AVAILABLE:
        print("CRITICAL: MetaDrive is not installed in this environment.")
        return
 
    obs_dim = 259
    action_dim = 2
 
    agent = SACAgent(state_dim=obs_dim, action_dim=action_dim)
    buffer = ReplayBuffer(capacity=200000)
 
    config = {
        "map": "C",
        "use_render": False,
        "horizon": 200,
    }
 
    env = MetaDriveEnv(config=config)
    obs = env.reset()
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, info = obs
 
    episode_reward = 0.0
    episode_count = 0
    recent_episode_rewards = deque(maxlen=20)
 
    print(f"Initial observation shape: {np.asarray(obs).shape}")
    print(f"Training for {total_steps} steps | batch_size={batch_size} | "
          f"warmup={start_training_after} steps before updates begin\n")
 
    for step in range(1, total_steps + 1):
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
 
        buffer.push(obs, action, reward, next_obs, float(done))
        episode_reward += reward
        obs = next_obs
 
        # Only start updating once the buffer has enough transitions
        # to sample a real batch from.
        if len(buffer) >= start_training_after and step % update_every == 0:
            critic_loss, actor_loss, alpha_val = agent.update_parameters(buffer, batch_size)
            if critic_loss is not None and step % 500 == 0:
                print(f"[Step {step:06d}] critic_loss={critic_loss:.4f} "
                      f"actor_loss={actor_loss:.4f} alpha={alpha_val:.4f}")
 
        if done:
            episode_count += 1
            recent_episode_rewards.append(episode_reward)
            avg_recent = sum(recent_episode_rewards) / len(recent_episode_rewards)
            reason = "arrive_dest" if info.get("arrive_dest", False) else \
                     ("crash" if info.get("crash", False) or info.get("out_of_road", False) else "timeout")
            print(f"  Episode {episode_count:04d} finished at step {step:06d} | "
                  f"reason={reason} | episode_reward={episode_reward:.4f} | "
                  f"avg_last_20={avg_recent:.4f}")
 
            episode_reward = 0.0
            obs = env.reset()
            if isinstance(obs, tuple) and len(obs) == 2:
                obs, info = obs
 
    env.close()
    print("\nSUCCESS: training loop completed.")
    print(f"Total episodes: {episode_count}")
 
 
if __name__ == "__main__":
    run_sac_training()
 
