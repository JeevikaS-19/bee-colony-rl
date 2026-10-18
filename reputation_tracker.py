import math


class ReputationTracker:
    def __init__(self, agent_ids, alpha=0.1):
        """
        Reputation and caste tracker based on Temporal Polyethism.

        E_t is an exponentially weighted arrival rate over TERMINAL outcomes only:
            E <- (1 - alpha) * E + alpha * outcome,   outcome = 1 (arrive), 0 (crash)
        E stays in [0, 1] and starts at 0 (NOVICE). Horizon timeouts are not
        terminal outcomes: the caller must not call update() for them.

        Args:
            agent_ids (list): unique agent identifiers to track.
            alpha (float): EWMA weight on the newest terminal outcome. Fixed
                before comparison runs; not tuned on the baseline's outcomes.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.E = {aid: 0.0 for aid in agent_ids}
        self.alpha = alpha

        # Caste boundaries (E is an arrival rate, so 0.3 means ~30% recent arrivals)
        self.THRESHOLD_FORAGER = 0.3
        self.THRESHOLD_SCOUT = 0.8

        # Vigor levels
        self.NOVICE_VIGOR = 0.05  # approved: low, but below the FORAGER floor
        self.FORAGER_MIN = 0.1
        self.FORAGER_MAX = 0.8
        self.SCOUT_VIGOR = 0.95

    def update(self, agent_id, success: bool):
        """
        Fold one terminal outcome into E_t. success=True is an arrival,
        success=False is a crash. Returns the new E_t.
        """
        if agent_id not in self.E:
            raise KeyError(f"Agent ID '{agent_id}' is not registered in the tracker.")
        outcome = 1.0 if success else 0.0
        self.E[agent_id] = (1.0 - self.alpha) * self.E[agent_id] + self.alpha * outcome
        return self.E[agent_id]

    def caste(self, agent_id):
        """Temporal caste of an agent, from its E_t."""
        exp = self.E[agent_id]
        if exp < self.THRESHOLD_FORAGER:
            return "NOVICE"
        elif exp < self.THRESHOLD_SCOUT:
            return "FORAGER"
        else:
            return "SCOUT"

    def vigor(self, agent_id):
        """
        Broadcast vigor for an agent.
        NOVICE: fixed low vigor (listens and may warn, never writes success).
        FORAGER: tanh(2E), clipped to [FORAGER_MIN, FORAGER_MAX].
        SCOUT: fixed high vigor.
        """
        exp = self.E[agent_id]
        c_name = self.caste(agent_id)

        if c_name == "NOVICE":
            return self.NOVICE_VIGOR
        elif c_name == "FORAGER":
            return max(self.FORAGER_MIN, min(self.FORAGER_MAX, math.tanh(2.0 * exp)))
        else:
            return self.SCOUT_VIGOR


if __name__ == "__main__":
    # Self-test: checks the EWMA arithmetic and the caste/vigor boundaries.
    t = ReputationTracker(agent_ids=["a"], alpha=0.1)

    # Four arrivals in a row: E = 1 - 0.9**4 = 0.3439 -> FORAGER
    for _ in range(4):
        t.update("a", True)
    assert abs(t.E["a"] - (1 - 0.9 ** 4)) < 1e-12, t.E["a"]
    assert t.caste("a") == "FORAGER", t.caste("a")
    assert 0.1 <= t.vigor("a") <= 0.8

    # Crashes pull E back down, and E never goes below 0
    for _ in range(50):
        t.update("a", False)
    assert t.E["a"] >= 0.0 and t.caste("a") == "NOVICE", (t.E["a"], t.caste("a"))
    assert t.vigor("a") == 0.05

    # Long run of arrivals approaches 1 and reaches SCOUT
    for _ in range(60):
        t.update("a", True)
    assert t.E["a"] <= 1.0 and t.caste("a") == "SCOUT", (t.E["a"], t.caste("a"))
    assert t.vigor("a") == 0.95

    # A constant 20% arrival rate settles near 0.2 -> still NOVICE (below 0.3)
    t2 = ReputationTracker(agent_ids=["b"], alpha=0.1)
    for i in range(2000):
        t2.update("b", i % 5 == 0)
    assert abs(t2.E["b"] - 0.2) < 0.05 and t2.caste("b") == "NOVICE", t2.E["b"]

    try:
        ReputationTracker(agent_ids=["c"], alpha=0.0)
        raise AssertionError("alpha=0 should be rejected")
    except ValueError:
        pass

    print("ReputationTracker self-test passed (EWMA, castes, vigor).")
