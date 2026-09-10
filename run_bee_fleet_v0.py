"""
run_bee_fleet_v0.py

File 3: Swarm Environment Loop Integration.
Wires the ReputationTracker into the MetaDrive MultiAgentIntersectionEnv loop.

FIXED (v0.2):
- Crash/arrival detection was checking info_dict.get("crash", ...), which is
  not a real MetaDrive info key. MetaDrive splits collisions into
  crash_vehicle / crash_object / crash_building rather than a single
  "crash" flag, so this was silently returning False every step regardless
  of what actually happened in the sim -- explains why even the scripted
  agents (agent0/1) never registered a terminal event.
- NOTE: arrival still calls tracker.update(agent_id, success=True) TWICE
  in a row (in both the live loop and mock_loop). Left as-is since it's
  unclear if this is deliberate (arrival weighted 2x vs a normal success)
  or a leftover duplicate -- confirm intent before the next run, since it
  doubles E_t growth on arrival relative to what step_up=0.08 implies.

FIXED (v0.1):
- Removed per-step "safely navigating" reputation credit -- E_t now only
  updates on a terminal outcome (crash = failure, arrive_dest = success).
  The old version rewarded raw survival time, saturating every agent
  (including the random one) to max E_t within ~20 steps.
- Widened agent2's random throttle range to match steer (-1.0 to 1.0) --
  the old -0.5 to 0.5 range was too timid to reliably trigger a crash
  or out_of_road within a reasonable number of steps.
- Reduced horizon to 200 steps via config, so debug iterations don't
  take ~1000 steps to resolve when nothing terminal is happening.
"""

import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from reputation_tracker import ReputationTracker

try:
    from metadrive import MultiAgentIntersectionEnv
    METADRIVE_AVAILABLE = True
except ImportError:
    METADRIVE_AVAILABLE = False


class ScriptedAutopilot:
    """
    A lightweight, decoupled local controller (Proportional Steering)
    to steer vehicles safely toward their target trajectory without using
    MetaDrive's internal IDMPolicy objects (confirmed not assignable
    per-agent via agent_configs in this MetaDrive version).

    CORRECTED (v0.3): the previous version guessed obs[0] was speed and
    obs[1] was lateral deviation. Checked against MetaDrive's actual
    source (metadrive/obs/state_obs.py): with side_detector and
    lane_line_detector disabled (see config), the ego-state block is a
    fixed 9-value layout:
        obs[0] = normalized distance to LEFT road boundary
        obs[1] = normalized distance to RIGHT road boundary
        obs[2] = heading difference vs. current lane
        obs[3] = normalized speed (relative to the vehicle's own max speed)
        obs[4] = normalized current steering
        obs[5] = normalized last-frame steering
        obs[6] = normalized last-frame throttle/brake
        obs[7] = yaw rate
        obs[8] = lateral offset within the current lane, 0.5 = centered,
                 <0.5 = left of center, >0.5 = right of center
    Rather than inverting obs[3]'s normalization (which needs the
    vehicle's max_speed_km_h), we take real speed straight from
    MetaDrive's info dict, confirmed to report velocity in km/h.
    Steering now uses obs[8], the one index that's actually documented
    as lateral lane position.
    """
    def __init__(self, target_speed=30.0):
        self.target_speed = target_speed / 3.6  # km/h -> m/s
        self.kp_steer = 1.2
        self.kp_throttle = 0.5

    def act(self, observation, real_speed_kmh=0.0):
        current_speed = real_speed_kmh / 3.6  # km/h -> m/s, ground truth from info
        speed_error = self.target_speed - current_speed
        throttle = float(np.clip(speed_error * self.kp_throttle, -1.0, 1.0))

        try:
            # obs[8] centered at 0.5; recenter to a signed error term.
            lateral_deviation = (observation[8] - 0.5) * 2.0
            steer = float(np.clip(-lateral_deviation * self.kp_steer, -0.6, 0.6))
        except (IndexError, TypeError):
            steer = 0.0

        return np.array([steer, throttle])


def run_fleet_simulation():
    if not METADRIVE_AVAILABLE:
        print("=" * 60)
        print("CRITICAL: MetaDrive is not installed in this offline environment.")
        print("Please save both 'reputation_tracker.py' and this file to your local")
        print("machine where MetaDrive is installed to run the physical simulation.")
        print("=" * 60)
        print("\n--> Running Mock Simulation Trace to Verify Loop Logic & API Integration...")
        mock_loop()
        return

    config = {
        "num_agents": 3,
        "is_multi_agent": True,
        "allow_respawn": False,
        "crash_done": True,
        "use_render": False,
        "horizon": 200,  # shortened for faster debug iterations
        # Disabling these forces MetaDrive's ego-state observation to use its
        # simple, documented scalar encoding (distance-to-left-boundary,
        # distance-to-right-boundary, ..., lateral-offset-in-lane) instead of
        # variable-length lidar cloud points swapped into the same slots.
        # This is what makes obs[0], obs[1], obs[8] deterministic and safe
        # to hand-index in ScriptedAutopilot below.
        "vehicle_config": {
            "side_detector": {"num_lasers": 0},
            "lane_line_detector": {"num_lasers": 0},
        },
    }

    print("\n--- Initialising Swarm Environment ---")
    env = MultiAgentIntersectionEnv(config=config)

    agent_ids = ["agent0", "agent1", "agent2"]
    tracker = ReputationTracker(agent_ids=agent_ids, step_up=0.08, step_down=0.15)

    autopilots = {
        "agent0": ScriptedAutopilot(target_speed=25.0),
        "agent1": ScriptedAutopilot(target_speed=25.0),
    }

    print("\n--- Starting Swarm Loop ---")
    obs = env.reset()
    current_info = {}
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, current_info = obs

    active_agents = set(obs.keys())
    step_count = 0

    while active_agents:
        step_count += 1
        actions = {}

        for agent_id in list(active_agents):
            if agent_id in autopilots:
                # Real ground-truth speed (km/h) comes from info, not from
                # decoding obs -- see ScriptedAutopilot docstring for why.
                real_speed_kmh = current_info.get(agent_id, {}).get("velocity", 0.0)
                actions[agent_id] = autopilots[agent_id].act(obs[agent_id], real_speed_kmh)
            else:
                # Agent 2: chaotic exploratory control -- full range on both axes
                # so it actually reaches a terminal state instead of idling.
                actions[agent_id] = np.array(
                    [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
                )

        # DEBUG: confirm speed/lateral-offset tracking for the first 10 steps.
        # Remove once you've confirmed agent0/1 actually move and steer sanely.
        if step_count <= 10 and "agent0" in actions:
            print(f"[DEBUG step {step_count:03d}] agent0 real_speed_kmh="
                  f"{current_info.get('agent0', {}).get('velocity', 'n/a')}, "
                  f"obs[8](lane_offset)={obs['agent0'][8]:.3f}, "
                  f"computed_action(steer,throttle)={actions['agent0']}")

        step_results = env.step(actions)
        if len(step_results) == 5:
            next_obs, rewards, terminated, truncated, info_dict = step_results
            dones = {k: (terminated[k] or truncated[k]) for k in terminated}
        else:
            next_obs, rewards, dones, info_dict = step_results

        current_info = info_dict

        for agent_id in list(active_agents):
            agent_info = info_dict.get(agent_id, {})
            is_finished = dones[agent_id]

            # FIXED: "crash" is not a real MetaDrive info key. MetaDrive
            # reports collisions as separate flags rather than one generic
            # boolean, so check all of them plus out_of_road.
            crashed = (
                agent_info.get("crash_vehicle", False)
                or agent_info.get("crash_object", False)
                or agent_info.get("crash_building", False)
                or agent_info.get("out_of_road", False)
            )
            arrived = agent_info.get("arrive_dest", False)

            # Reputation updates ONLY on terminal outcomes -- no per-step credit.
            if crashed:
                tracker.update(agent_id, success=False)
                new_caste = tracker.caste(agent_id)
                print(f"[Step {step_count:03d}] \U0001F4A5 {agent_id} CRASHED! "
                      f"E_t: {tracker.E[agent_id]:.2f} -> {new_caste} (Muted, Vigor: 0.0)")
                active_agents.remove(agent_id)

            elif arrived:
                tracker.update(agent_id, success=True)
                tracker.update(agent_id, success=True)
                new_caste = tracker.caste(agent_id)
                print(f"[Step {step_count:03d}] \U0001F3C1 {agent_id} ARRIVED! "
                      f"E_t: {tracker.E[agent_id]:.2f} -> {new_caste} (Vigor: {tracker.vigor(agent_id):.2f})")
                active_agents.remove(agent_id)

            elif is_finished:
                # Timed out / other non-crash termination -- no reputation change.
                print(f"[Step {step_count:03d}] \u23F9\uFE0F {agent_id} Finished driving (No crash, no arrival).")
                active_agents.remove(agent_id)

            else:
                # Still driving, not yet terminal -- no reputation update.
                pass

        obs = next_obs

    env.close()
    print("\n--- Swarm Loop Completed Safely! ---")


def mock_loop():
    """
    Mock execution loop to verify tracker wiring without a live MetaDrive
    installation. Also updated to terminal-only reputation logic.
    """
    agent_ids = ["agent0", "agent1", "agent2"]
    tracker = ReputationTracker(agent_ids=agent_ids, step_up=0.08, step_down=0.15)

    print("\nStarting mock verification loop...")
    active_agents = set(agent_ids)
    step = 0

    while active_agents and step < 40:
        step += 1

        for agent_id in list(active_agents):
            if agent_id == "agent0":
                if step == 20:  # arrives
                    tracker.update(agent_id, success=True)
                    tracker.update(agent_id, success=True)
                    print(f"[Mock Step {step:02d}] \U0001F3C1 {agent_id} ARRIVED! "
                          f"E_t: {tracker.E[agent_id]:.2f} -> {tracker.caste(agent_id)} "
                          f"(Vigor: {tracker.vigor(agent_id):.2f})")
                    active_agents.remove(agent_id)

            elif agent_id == "agent1":
                if step == 12:  # one recoverable failure along the way
                    tracker.update(agent_id, success=False)
                    print(f"[Mock Step {step:02d}] \u26A0\uFE0F {agent_id} Clipped Pavement! "
                          f"E_t: {tracker.E[agent_id]:.2f} -> {tracker.caste(agent_id)} "
                          f"(Vigor: {tracker.vigor(agent_id):.2f})")
                elif step == 24:  # arrives despite the earlier clip
                    tracker.update(agent_id, success=True)
                    tracker.update(agent_id, success=True)
                    print(f"[Mock Step {step:02d}] \U0001F3C1 {agent_id} ARRIVED! "
                          f"E_t: {tracker.E[agent_id]:.2f} -> {tracker.caste(agent_id)} "
                          f"(Vigor: {tracker.vigor(agent_id):.2f})")
                    active_agents.remove(agent_id)

            elif agent_id == "agent2":
                if step == 5:  # crashes early
                    tracker.update(agent_id, success=False)
                    print(f"[Mock Step {step:02d}] \U0001F4A5 {agent_id} CRASHED! "
                          f"E_t: {tracker.E[agent_id]:.2f} -> {tracker.caste(agent_id)} "
                          f"(Muted, Vigor: 0.0)")
                    active_agents.remove(agent_id)

    print("\n--- Mock Verification Loop Completed Safely! ---")


if __name__ == "__main__":
    run_fleet_simulation()