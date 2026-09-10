# Project Bee — Bee-Colony-Inspired Multi-Agent RL

A training-efficiency mechanism for RL, built as a follow-up to the [RL Self-Driving Car](https://github.com/JeevikaS-19/rl-self-driving-car) project after single-agent SAC training turned out to not be sample-efficient enough per step.

## Problem

Single-agent RL means one agent has to rediscover everything on its own, one mistake at a time. Inspired by China's V2X smart-infrastructure approach to connected autonomous vehicles, the question was: could many agents exploring in parallel, and sharing what they learn, make training itself more sample-efficient? And more specifically: multi-agent RL communication is famously hard to get right, can a bee-colony/hive structure actually bridge that gap?

**Important scope note:** this is a training-phase mechanism only, not real-time multi-vehicle coordination. At deployment, only one trained policy needs to drive, this isn't a coordinated fleet-driving product.

## Solution

Around 10 "bees" train in parallel, independent MetaDrive environments, no real-time physical interaction between them. All bees share **one policy network** (parameter sharing, similar to Ape-X/IMPALA-style distributed RL, not separate models per bee).

When a bee successfully pulls off something hard, like a difficult turn, it leaves a pheromone trace behind. Other bees can pick up on that trace and learn the same thing faster, instead of rediscovering it independently from scratch.

Two supporting ideas carried over from the bee metaphor:
- **Waggle-dance guidance**: successful traces are encoded as a fixed message format (direction, distance, confidence), not a free-form learned embedding.
- **Temporal polyethism (experience-graded teaching)**: a staged reputation system. Inexperienced bees mostly explore and listen. Bees that prove reliable start guiding others, weighted by earned credibility, not just vote counts.

## Architecture

1. **N independent MetaDrive environments**, run via true Python multiprocessing (since MetaDrive is Python-only, multiprocessing is the actual fix for the GIL, not multi-agent co-located envs).
2. **One shared SAC policy network** across all workers, fresh and untrained (flat 259-dim observation space, different from the main self-driving project's CNN+attention setup, so no checkpoint reuse between the two).
3. **Reputation/vigor tracker** — a discontinuous NOVICE / FORAGER / SCOUT caste system, with hard experience-index thresholds at 0.3 and 0.8. Experience only rises on successfully completing hard trajectories, not on raw step count. Discontinuous bands were chosen deliberately over a smooth curve, to block noise.
4. **Shared pheromone table** — a spatial hash on 2m grid cells. Each cell is overwritten only if the new trace has higher vigor (to bound memory, not append-only). A successful run deposits a "turning guide" trace along its path; a crash deposits a single warning trace at the crash point.
5. **Reward-side bias, not action-blending** — being near a pheromone trace shapes the *reward*, not the action itself. This is deliberate: the trained policy shouldn't end up depending on guidance that won't exist when it's driving alone at test time.
6. **Staged activation** — pheromone reading only switches on after a baseline period (~100k-250k steps), so each network has its own grounding before it starts leaning on guidance from others.

Quorum-sensing/consensus was originally planned as a separate mechanism for breaking deadlocks, but was dropped: route distribution and noise-robustness turned out to be handled fine within the vigor-weighted waggle-dance design alone.

## Status

In progress. Built and verified so far:
- Reputation/vigor tracker
- True parallel multiprocessing across independent MetaDrive workers (Windows spawn-safe)
- Full single-worker SAC implementation (actor, twin critic, replay buffer, auto-tuned entropy)
- Pheromone table structure

Currently: running the first full single-worker SAC training loop alongside finishing the pheromone table, before wiring the two together into the full multi-bee system.
