import math

class ReputationTracker:
    def __init__(self, agent_ids, step_up=0.05, step_down=0.10, e_max=1.0):
        """
        Reputation and Caste Tracker based on Temporal Polyethism.
        Tracks agent reliability and maps it to communication castes.
        
        Args:
            agent_ids (list): List of unique agent identifiers to track.
            step_up (float): Incremental reputation increase on successful step.
            step_down (float): Reputation penalty on safety violation/failure (Asymmetric Decay).
            e_max (float): Bounded maximum experience score.
        """
        self.E = {aid: 0.0 for aid in agent_ids}
        self.e_max = e_max
        self.step_up = step_up
        self.step_down = step_down
        
        # Caste Boundaries
        self.THRESHOLD_FORAGER = 0.3
        self.THRESHOLD_SCOUT = 0.8

    def update(self, agent_id, success: bool):
        """
        Updates an agent's experience score (E_t) based on performance.
        Safety violations actively penalise (decay) experience.
        """
        if agent_id not in self.E:
            raise KeyError(f"Agent ID '{agent_id}' is not registered in the tracker.")
            
        if success:
            # Gradually accumulate reputation
            self.E[agent_id] = min(self.e_max, self.E[agent_id] + self.step_up)
        else:
            # Actively decay reputation on failure (allows "tanking" below threshold)
            self.E[agent_id] = max(0.0, self.E[agent_id] - self.step_down)
            
        return self.E[agent_id]

    def caste(self, agent_id):
        """
        Returns the temporal caste of an agent based on its experience.
        """
        exp = self.E[agent_id]
        if exp < self.THRESHOLD_FORAGER:
            return "NOVICE"
        elif exp < self.THRESHOLD_SCOUT:
            return "FORAGER"
        else:
            return "SCOUT"

    def vigor(self, agent_id):
        """
        Returns the V2X broadcast vigor (v).
        Novices are completely muted (v = 0.0).
        Foragers scale continuously: v = tanh(E_t)
        Scouts have high locked minimum authority.
        """
        exp = self.E[agent_id]
        c_name = self.caste(agent_id)
        
        if c_name == "NOVICE":
            return 0.0
        elif c_name == "FORAGER":
            # Continuous scaling normalized to [0.1, 0.8] range
            return max(0.1, min(0.8, math.tanh(2.0 * exp)))
        else: # SCOUT
            # High baseline authority locked in
            return 0.95


if __name__ == "__main__":
    # Self-test block
    tracker = ReputationTracker(agent_ids=["agent_0", "agent_1", "agent_2"])
    
    print("=== Standalone Reputation Tracker Test ===")
    print("\n--- 1. Testing Agent 0 (Continuous Success) ---")
    for step in range(1, 21):
        tracker.update("agent_0", success=True)
        print(f"Step {step:02d} | Experience: {tracker.E['agent_0']:.2f} | Caste: {tracker.caste('agent_0'):<7} | Vigor: {tracker.vigor('agent_0'):.2f}")
        
    print("\n--- 2. Testing Agent 1 (Intermittent Success & Failures) ---")
    script = [True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True]
    for step, outcome in enumerate(script, 1):
        tracker.update("agent_1", success=outcome)
        print(f"Step {step:02d} | Outcome: {'SUCCESS' if outcome else 'FAILURE':<7} | Experience: {tracker.E['agent_1']:.2f} | Caste: {tracker.caste('agent_1'):<7} | Vigor: {tracker.vigor('agent_1'):.2f}")

    print("\n--- 3. Testing Agent 2 (Catastrophic Crashes & Tanking) ---")
    # Accumulate some experience first
    for step in range(1, 6):
        tracker.update("agent_2", success=True)
    print(f"Initial: Experience = {tracker.E['agent_2']:.2f}, Caste = {tracker.caste('agent_2')}")
    
    for step in range(1, 6):
        tracker.update("agent_2", success=False)
        print(f"Crash {step:02d} | Experience: {tracker.E['agent_2']:.2f} | Caste: {tracker.caste('agent_2'):<7} | Vigor: {tracker.vigor('agent_2'):.2f}")