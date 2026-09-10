#what this file does, sanity check for the multi agent metadrive
import time 
import random 
import numpy as np

try: 
    from metadrive import MultiAgentIntersectionEnv    
    ENV_CLASS = MultiAgentIntersectionEnv
    print("Successfully imported multi agent")
except ImportError:
    try: 
        from metadrive.envs.multi_agent_env import MultiAgentEnv
        ENV_CLASS = MultiAgentEnv
        print("fallback to generic multiagentenv")
    except ImportError as e:
        print("Critical: metadrive is not installed or accessible")
        raise e

def run_sanity_check():
    config = {
        "num_agents": 3,
        "is_multi_agent": True,
        "allow_respawn": False,
        "use_render": True, 
        "crash_done": True,
    }
    print("\n--- Initialising Environment ---")
    try:
        # If fallback, we pass a map/scenario key, otherwise instantiate scenario class
        if ENV_CLASS.__name__ == "MultiAgentEnv":
            env = ENV_CLASS(config=dict(config, map="I")) # "I" for Intersection
        else:
            env = ENV_CLASS(config=config)
        print(f"Environment instantiated successfully: {ENV_CLASS.__name__}")
    except Exception as e:
        print(f"Failed to instantiate environment: {e}")
        return

    # Keep track of active agents during the episode
    active_agents = set()
    step_count = 0
    start_time = time.time()

    try:
        # 2. Reset Environment and verify observation structure
        print("\n--- Resetting Environment ---")
        reset_results = env.reset()
        
        # Handle unified gymnasium API return differences (obs or obs, info)
        if isinstance(reset_results, tuple) and len(reset_results) == 2:
            obs, info = reset_results
        else:
            obs = reset_results

        print(f"Reset returned observations for agents: {list(obs.keys())}")
        for agent_id in obs.keys():
            active_agents.add(agent_id)
            print(f"  -> {agent_id} observation shape: {obs[agent_id].shape}")

        print("\n--- Running Step Loop (Simulating Exploration) ---")
        
        # Run until all agents are finished (or we hit a safety limit)
        max_steps = 1000
        while active_agents and step_count < max_steps:
            step_count += 1
            
            # Generate random baseline actions for actively driving vehicles
            actions = {}
            for agent_id in active_agents:
                # Sample continuous steering (-1 to 1) and throttle/brake (-1 to 1)
                # We can sample directly from the agent's action space
                actions[agent_id] = env.action_space[agent_id].sample()

            # Execute step
            step_results = env.step(actions)
            
            # Dynamically unpack step returns (unifies older 4-tuple vs newer 5-tuple APIs)
            if len(step_results) == 5:
                next_obs, rewards, terminated, truncated, info = step_results
                # Combine terminal states for active agent tracking
                dones = {k: (terminated[k] or truncated[k]) for k in terminated}
            else:
                next_obs, rewards, dones, info = step_results

            # 3. Asynchronous Termination Checks
            newly_done = []
            for agent_id in list(active_agents):
                # Verify we can index cleanly by Agent ID
                r = rewards[agent_id]
                done = dones[agent_id]
                
                # Check for individual termination events
                if done:
                    newly_done.append(agent_id)
                    active_agents.remove(agent_id)
                    print(f"[Step {step_count:03d}] Agent '{agent_id}' finished. Reward: {r:.2f}")

            # Verify global termination signal
            global_done = dones.get("__all__", False)
            if global_done or not active_agents:
                print(f"\n[Step {step_count:03d}] Global Termination '__all__' triggered.")
                break

        # 4. Success Performance Report
        end_time = time.time()
        elapsed = end_time - start_time
        steps_per_sec = step_count / elapsed if elapsed > 0 else 0

        print("\n=== Baseline Sanity Check: SUCCESS ===")
        print(f"Total Steps Executed : {step_count}")
        print(f"Total Time Elapsed    : {elapsed:.3f} seconds")
        print(f"Raw Execution Speed  : {steps_per_sec:.2f} steps/second")
        print(f"All agents cleaned up asynchronously without training loop crashes.")
        print("=======================================")

    except Exception as e:
        print(f"\nCRITICAL SANITY RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment closed cleanly.")


if __name__ == "__main__":
    run_sanity_check()

