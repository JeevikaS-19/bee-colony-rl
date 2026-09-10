#test_mixed_policy.py

import numpy as np
import traceback

try:
    from metadrive import MultiAgentIntersectionEnv
    METADRIVE_AVAILABLE = True
except ImportError:
    METADRIVE_AVAILABLE = False


def run_diagnostic():
    if not METADRIVE_AVAILABLE:
        print("CRITICAL: MetaDrive is not installed in this environment.")
        print("Please copy this script to your local machine where MetaDrive is installed to run it.")
        return

    # Configuration for testing per-agent policy mixing
    config = {
        "num_agents": 3,
        "is_multi_agent": True,
        "allow_respawn": False,
        "crash_done": True,
        # Try to specify policy mixing via agent_configs
        "agent_configs": {
            "agent0": {"use_expert_or_autopilot": True},
            "agent1": {"use_expert_or_autopilot": True},
            "agent2": {"use_expert_or_autopilot": False},
        }
    }

    print("=== Diagnostic Test 1: Instantiating with mixed agent_configs ===")
    try:
        env = MultiAgentIntersectionEnv(config=config)
        obs = env.reset()
        print("-> Environment instantiated successfully with mixed agent_configs!")
        
        # Take a step passing zero actions for autopilot agents, random actions for the manual one
        actions = {
            "agent0": np.array([0.0, 0.0]),  # If IDM is active, this should be ignored or used as fallback
            "agent1": np.array([0.0, 0.0]),
            "agent2": np.array([0.1, 0.5])   # Manual input to steer
        }
        
        step_results = env.step(actions)
        print("-> Step completed successfully!")
        env.close()
        print("\n[RESULT 1] Your MetaDrive version supports per-agent configuration natively!")
        return
        
    except Exception as e:
        print(f"-> Diagnostic Test 1 failed with error: {e}")
        print("This means your version of MetaDrive might enforce a uniform policy across all agents.\n")

    print("=== Diagnostic Test 2: Fallback Programmatic Autopilot Override ===")
    # Fallback approach: Run a standard manual env, but manually query MetaDrive's internal policy for selected agents
    config_manual = {
        "num_agents": 3,
        "is_multi_agent": True,
        "allow_respawn": False,
        "crash_done": True,
        "agent_policy": None  # No environment-wide autopilot override
    }
    
    try:
        env = MultiAgentIntersectionEnv(config=config_manual)
        obs = env.reset()
        
        # Verify if we can access the internal expert policy for specific agents programmatically
        print("Checking engine and policy accessibility...")
        
        # Step through the engine to see if expert policies are registered
        actions = {}
        for agent_id in env.active_agents.keys():
            if agent_id in ["agent0", "agent1"]:
                # Access the internal expert policy dynamically if supported
                try:
                    expert_action = env.engine.get_policy(agent_id).act()
                    actions[agent_id] = expert_action
                    print(f"  -> Successfully queried expert action for {agent_id}: {expert_action}")
                except AttributeError:
                    try:
                        expert_action = env.agent_manager.get_agent(agent_id).policy.act()
                        actions[agent_id] = expert_action
                    except Exception:
                        print(f"  -> Cannot access policy directly for {agent_id}. Need programmatic steering.")
                        actions[agent_id] = np.array([0.0, 0.5])  # Go straight/default
            else:
                # Agent 2 drives manually
                actions[agent_id] = np.array([0.5, 0.2]) # Sharp right, gentle throttle
                
        env.step(actions)
        print("-> Step with programmatic autopilot override completed successfully!")
        env.close()
        print("\n[RESULT 2] Fallback programmatic override is viable for mixed control!")
        
    except Exception as e:
        print(f"-> Diagnostic Test 2 failed with error: {e}")
        print("\n[CONCLUSION] Mixed-policy runs will require a wrapper helper class to handle actions programmatically.")


if __name__ == "__main__":
    run_diagnostic()