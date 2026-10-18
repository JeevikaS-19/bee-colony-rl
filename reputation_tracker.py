import math


class ReputationTracker:
    def __init__(self, agent_ids, smoothing=0.1, hysteresis=0.05):
        """
        Reputation and caste tracker based on Temporal Polyethism.

        E_t is an exponentially weighted arrival rate over TERMINAL episode ends:
            E <- (1 - smoothing) * E + smoothing * outcome
        outcome = 1 for an arrival, 0 for anything else (crash, out-of-road,
        or horizon timeout: a timeout is a failure to arrive).
        E stays in [0, 1] and starts at 0 (NOVICE).

        Castes use hysteresis: a bee is promoted at the threshold but only
        demoted once E falls `hysteresis` below it. This stops a bee whose
        true arrival rate sits near a threshold from flickering between castes.

        Args:
            agent_ids (list): unique agent identifiers to track.
            smoothing (float): EWMA weight on the newest outcome, in (0, 1].
                Fixed before comparison runs; not tuned on the baseline's outcomes.
            hysteresis (float): margin below a threshold before demotion, >= 0.
        """
        if not 0.0 < smoothing <= 1.0:
            raise ValueError(f"smoothing must be in (0, 1], got {smoothing}")
        if hysteresis < 0.0:
            raise ValueError(f"hysteresis must be >= 0, got {hysteresis}")
        self.E = {aid: 0.0 for aid in agent_ids}
        self.smoothing = smoothing
        self.hysteresis = hysteresis
        self._caste = {aid: "NOVICE" for aid in agent_ids}

        # Caste thresholds (E is an arrival rate, so 0.3 means ~30% recent arrivals)
        self.THRESHOLD_FORAGER = 0.3
        self.THRESHOLD_SCOUT = 0.8

        # Vigor levels
        self.NOVICE_VIGOR = 0.05  # low, below the FORAGER floor
        self.FORAGER_MIN = 0.1
        self.FORAGER_MAX = 0.8
        self.SCOUT_VIGOR = 0.95

    def update(self, agent_id, success: bool):
        """
        Fold one episode end into E_t. success=True only for an arrival;
        crashes and horizon timeouts pass success=False. Returns the new E_t.
        """
        if agent_id not in self.E:
            raise KeyError(f"Agent ID '{agent_id}' is not registered in the tracker.")
        outcome = 1.0 if success else 0.0
        self.E[agent_id] = (1.0 - self.smoothing) * self.E[agent_id] + self.smoothing * outcome
        self._caste[agent_id] = self._next_caste(agent_id)
        return self.E[agent_id]

    def _next_caste(self, agent_id):
        exp = self.E[agent_id]
        current = self._caste[agent_id]
        h = self.hysteresis
        if current == "SCOUT":
            if exp >= self.THRESHOLD_SCOUT - h:
                return "SCOUT"
            current = "FORAGER"  # demoted one level; check FORAGER band below
        if current == "FORAGER":
            if exp >= self.THRESHOLD_SCOUT:
                return "SCOUT"
            if exp >= self.THRESHOLD_FORAGER - h:
                return "FORAGER"
            return "NOVICE"
        # current == NOVICE
        if exp >= self.THRESHOLD_SCOUT:
            return "SCOUT"
        if exp >= self.THRESHOLD_FORAGER:
            return "FORAGER"
        return "NOVICE"

    def caste(self, agent_id):
        """Current caste of an agent (updated on each update() call)."""
        return self._caste[agent_id]

    def vigor(self, agent_id):
        """
        Broadcast vigor for an agent's current caste.
        NOVICE: fixed low vigor (may warn, never writes success).
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
    # Self-test: EWMA arithmetic, promotion, hysteresis, and vigor.
    t = ReputationTracker(agent_ids=["a"], smoothing=0.1, hysteresis=0.05)

    # Four arrivals in a row: E = 1 - 0.9**4 = 0.3439 -> promoted to FORAGER
    for _ in range(4):
        t.update("a", True)
    assert abs(t.E["a"] - (1 - 0.9 ** 4)) < 1e-12, t.E["a"]
    assert t.caste("a") == "FORAGER", t.caste("a")
    assert 0.1 <= t.vigor("a") <= 0.8

    # Hysteresis: a crash drops E to 0.309 (still >= 0.25) -> stays FORAGER
    t.update("a", False)
    assert t.E["a"] >= 0.25 and t.caste("a") == "FORAGER", (t.E["a"], t.caste("a"))
    # Keep crashing until E < 0.25 -> demoted to NOVICE
    while t.E["a"] >= 0.25:
        t.update("a", False)
    assert t.caste("a") == "NOVICE", (t.E["a"], t.caste("a"))
    assert t.vigor("a") == 0.05

    # Timeouts count as failures: a bee that never crashes but never arrives decays
    t_timeout = ReputationTracker(agent_ids=["x"], smoothing=0.1)
    for _ in range(4):
        t_timeout.update("x", True)
    for _ in range(30):
        t_timeout.update("x", False)  # a timeout
    assert t_timeout.E["x"] < 0.05, t_timeout.E["x"]

    # Long run of arrivals reaches SCOUT
    for _ in range(60):
        t.update("a", True)
    assert t.E["a"] <= 1.0 and t.caste("a") == "SCOUT", (t.E["a"], t.caste("a"))
    assert t.vigor("a") == 0.95

    # Constant 20% arrival rate: E settles near 0.2, below the 0.25 demotion
    # line for a bee already NOVICE, so it stays NOVICE and does not flicker up.
    t2 = ReputationTracker(agent_ids=["b"], smoothing=0.1, hysteresis=0.05)
    castes = []
    for i in range(2000):
        t2.update("b", i % 5 == 0)
        castes.append(t2.caste("b"))
    assert abs(t2.E["b"] - 0.2) < 0.1, t2.E["b"]
    assert "FORAGER" not in castes[500:] or castes[500:].count("FORAGER") < 50, "flicker"

    for bad in (dict(smoothing=0.0), dict(hysteresis=-0.1)):
        try:
            ReputationTracker(agent_ids=["c"], **bad)
            raise AssertionError(f"{bad} should be rejected")
        except ValueError:
            pass

    print("ReputationTracker self-test passed (EWMA, hysteresis, timeouts, vigor).")
