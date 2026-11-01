"""
Stall rescue ("Layer 1: adaptive curriculum + diversity"). Learner-side, pure Python/numpy.

Why a separate module and not shaping_calculator.py: the diversity and exploration terms
need what the OTHER bees did, and only the learner sees all bees. Workers are separate
processes. So the learner computes these extra reward terms when an episode arrives and
adds them to the transitions before they enter the replay buffer. shaping_calculator.py
(shared with the single-bee loop) is left untouched.

These terms are NOT potential-based. They change the reward the policy optimises, and they
are only active after a stall has been flagged. Any arm that uses them is a different
experiment from the plain shaping arm and needs its own no-shaping control.

Terms (all per step, all zero until the stall flag is on; the flag latches):
  forward     +forward_scale * euclidean distance moved this step (metres)
  diversity   -diversity_scale * sim, for each OTHER bee whose last action in the same
              discretised state was within the similarity threshold (past episodes only)
  exploration +explore_scale / (visits + 1), per bee, visits counted on the discretised state
  The sum is clipped to [-clip, +clip] per step.

Differences from the written spec (and why):
  * similarity is L2-based, not cosine. Stall actions are near-zero vectors, and cosine
    ignores magnitude: [0.01, 0.01] and [0.9, 0.9] score 1.0.
  * stall mode defaults to "timeout" (last N post-warmup episodes ALL timeouts). The
    literal "zero arrivals in N episodes" rule is true in every healthy run for the first
    ~150 episodes (the baselines first arrived at episode 147-150). Use mode="arrival"
    for the literal rule.
  * only episodes after warmup count toward stall detection.
  * history uses an OrderedDict with FIFO eviction. dict.popitem() removes the NEWEST item.
  * diversity_scale default is 0.005, not 0.2. Measured on a creeping episode (500 steps, 7 m,
    three bees copying): 0.2 costs about -200 per episode, 0.02 about -20 (worse than a crash,
    about -17), 0.005 about -5. The creeping return is about +6.5.
"""
import math
from collections import OrderedDict, deque

import numpy as np

ACTION_MAX_DIST = 2.0 * math.sqrt(2.0)  # largest L2 distance between two actions in [-1, 1]^2


def action_similarity(a, b):
    """1.0 for identical actions, 0.0 for opposite corners. L2-based, magnitude-aware."""
    d = float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    return max(0.0, 1.0 - d / ACTION_MAX_DIST)


class StallRescue:
    def __init__(self, num_bees, warmup_steps=0, stall_window=20, stall_mode="timeout",
                 forward_scale=0.5, diversity_scale=0.005, explore_scale=0.05,
                 sim_threshold=0.90, cell=2.0, heading_bins=8, speed_bin_kmh=5.0,
                 history_cap=500, clip=1.0):
        assert stall_mode in ("timeout", "arrival")
        self.num_bees = num_bees
        self.warmup_steps = warmup_steps
        self.stall_window = stall_window
        self.stall_mode = stall_mode
        self.forward_scale = forward_scale
        self.diversity_scale = diversity_scale
        self.explore_scale = explore_scale
        self.sim_threshold = sim_threshold
        self.cell = cell
        self.heading_bins = heading_bins
        self.speed_bin_kmh = speed_bin_kmh
        self.history_cap = history_cap
        self.clip = clip

        self.recent = deque(maxlen=stall_window)  # terminal strings, post-warmup only
        self.active = False                       # latched: never unflagged
        self.activated_at = None                  # global step when it flipped
        self.action_history = {i: OrderedDict() for i in range(num_bees)}  # key -> last action
        self.visits = {i: {} for i in range(num_bees)}

    # ---- stall detection -------------------------------------------------------------
    def observe_episode(self, terminal, global_step):
        """terminal in {'arrive_dest','timeout','out_of_road','crash_*'}. Returns True the
        moment the flag flips on."""
        if global_step < self.warmup_steps:
            return False
        self.recent.append(terminal)
        if self.active or len(self.recent) < self.stall_window:
            return False
        if self.stall_mode == "timeout":
            stalled = all(t == "timeout" for t in self.recent)
        else:
            stalled = all(t != "arrive_dest" for t in self.recent)
        if stalled:
            self.active = True
            self.activated_at = global_step
            return True
        return False

    # ---- state discretisation (same for every bee) -----------------------------------
    def state_key(self, pos, heading, speed_kmh):
        hb = int(math.floor(((heading % (2 * math.pi)) / (2 * math.pi)) * self.heading_bins)) % self.heading_bins
        return (int(math.floor(pos[0] / self.cell)), int(math.floor(pos[1] / self.cell)),
                hb, int(speed_kmh // self.speed_bin_kmh))

    # ---- per-episode bonuses ---------------------------------------------------------
    def episode_bonuses(self, worker_id, start_state, path, actions):
        """
        start_state: (pos, heading, speed) BEFORE the first action.
        path:        per step (pos, heading, steer, speed) recorded AFTER that step.
        actions:     per step action taken, [steer, throttle].
        Step i is taken in the state before it: start_state for i=0, else path[i-1].
        Returns (per-step bonus array, diagnostics). History and visit counts are always
        updated (so the stall cells are already 'visited' when the flag flips); the bonus
        is zero while the flag is off. Only PAST episodes of other bees are compared.
        """
        n = len(actions)
        pre = [start_state] + [(p[0], p[1], p[3]) for p in path[:-1]]
        post_pos = [p[0] for p in path]
        fwd = np.zeros(n)
        div = np.zeros(n)
        exp = np.zeros(n)
        others = [j for j in range(self.num_bees) if j != worker_id]
        new_hist = []

        for i in range(n):
            pos, heading, speed = pre[i]
            key = self.state_key(pos, heading, speed)
            a = np.asarray(actions[i], dtype=float)

            if self.active:
                fwd[i] = self.forward_scale * math.hypot(post_pos[i][0] - pos[0], post_pos[i][1] - pos[1])
                for j in others:
                    other_a = self.action_history[j].get(key)
                    if other_a is not None:
                        sim = action_similarity(a, other_a)
                        if sim > self.sim_threshold:
                            div[i] += self.diversity_scale * sim
                exp[i] = self.explore_scale / (self.visits[worker_id].get(key, 0) + 1)

            self.visits[worker_id][key] = self.visits[worker_id].get(key, 0) + 1
            new_hist.append((key, a))

        # Update this bee's history AFTER scoring, so an episode never compares to itself.
        hist = self.action_history[worker_id]
        for key, a in new_hist:
            hist[key] = a
            hist.move_to_end(key)
            while len(hist) > self.history_cap:
                hist.popitem(last=False)  # oldest

        raw = fwd - div + exp
        total = np.clip(raw, -self.clip, self.clip)
        diag = {
            "forward_sum": float(fwd.sum()), "diversity_sum": float(div.sum()),
            "explore_sum": float(exp.sum()), "total_sum": float(total.sum()),
            "clipped_steps": int((np.abs(raw) > self.clip).sum()),
            "max_abs_step": float(np.abs(raw).max()) if n else 0.0,
        }
        return total, diag


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    def mk(n, x0=0.0, dx=1.0, speed=10.0):
        """A straight path of n steps along +x."""
        start = ((x0, 0.0), 0.0, speed)
        path = [((x0 + dx * (i + 1), 0.0), 0.0, 0.0, speed) for i in range(n)]
        return start, path

    # 1. Stall latch (timeout mode): 19 timeouts no flag, 20th flags, stays on after an arrival.
    r = StallRescue(3, warmup_steps=100, stall_window=20)
    for i in range(19):
        assert r.observe_episode("timeout", 200) is False
    assert not r.active
    assert r.observe_episode("timeout", 300) is True and r.active and r.activated_at == 300
    r.observe_episode("arrive_dest", 400)
    assert r.active, "flag must latch"

    # 2. Warmup episodes do not count; a crash in the window prevents a timeout-mode flag.
    r = StallRescue(3, warmup_steps=1000, stall_window=5)
    for _ in range(50):
        r.observe_episode("timeout", 10)
    assert not r.active and len(r.recent) == 0
    for t in ["timeout"] * 4 + ["out_of_road"] + ["timeout"] * 4:
        r.observe_episode(t, 2000)
    assert not r.active
    r.observe_episode("timeout", 2000)
    assert r.active  # last 5 are all timeouts

    # 3. Arrival mode: crashes count as 'no arrival'; one arrival blocks the flag.
    r = StallRescue(3, stall_window=3, stall_mode="arrival")
    for t in ["out_of_road", "arrive_dest", "out_of_road", "timeout"]:
        r.observe_episode(t, 10)
    assert not r.active
    r.observe_episode("out_of_road", 10)
    assert r.active

    # 4. Everything is zero while inactive, but history/visits are still recorded.
    r = StallRescue(2)
    s, p = mk(5)
    b, d = r.episode_bonuses(0, s, p, [[0.0, 0.5]] * 5)
    assert np.all(b == 0) and d["total_sum"] == 0
    assert len(r.action_history[0]) > 0 and sum(r.visits[0].values()) == 5

    # 5. Forward bonus = scale * distance (metres), exact.
    r = StallRescue(2, forward_scale=0.5, explore_scale=0.0, diversity_scale=0.0)
    r.active = True
    s, p = mk(4, dx=2.0)
    b, d = r.episode_bonuses(0, s, p, [[0.0, 0.5]] * 4)
    assert np.allclose(b, 0.5 * 2.0) and abs(d["forward_sum"] - 4.0) < 1e-9

    # 6. Diversity: bee 1 copying bee 0 in the same states is penalised; bee 0 (first) is not;
    #    a different action is not penalised; the first bee never compares to itself.
    r = StallRescue(2, forward_scale=0.0, explore_scale=0.0, diversity_scale=0.1, sim_threshold=0.9)
    r.active = True
    s, p = mk(5, dx=0.1, speed=2.0)  # creeping: stays in one 2 m cell
    b0, d0 = r.episode_bonuses(0, s, p, [[0.0, 0.1]] * 5)
    assert d0["diversity_sum"] == 0.0, "no history from others yet"
    b1, d1 = r.episode_bonuses(1, s, p, [[0.0, 0.1]] * 5)
    assert d1["diversity_sum"] > 0 and np.all(b1 < 0), "copying must cost"
    b1b, d1b = r.episode_bonuses(1, s, p, [[0.0, 0.9]] * 5)  # clearly different throttle
    assert d1b["diversity_sum"] == 0.0

    # 7. Cosine would have been wrong here; L2 similarity separates tiny from large actions.
    assert action_similarity([0.01, 0.01], [0.9, 0.9]) < 0.7
    assert action_similarity([0.2, 0.3], [0.2, 0.3]) == 1.0

    # 8. Exploration decays as 1/(visits+1) and is per bee.
    r = StallRescue(2, forward_scale=0.0, diversity_scale=0.0, explore_scale=1.0)
    r.active = True
    s, p = mk(1, dx=0.1, speed=2.0)
    vals = [r.episode_bonuses(0, s, p, [[0, 0]])[0][0] for _ in range(4)]
    assert np.allclose(vals, [1.0, 0.5, 1 / 3, 0.25]), vals
    assert np.isclose(r.episode_bonuses(1, s, p, [[0, 0]])[0][0], 1.0), "other bee starts fresh"

    # 9. Per-step clip.
    r = StallRescue(2, forward_scale=10.0, explore_scale=0.0, diversity_scale=0.0, clip=1.0)
    r.active = True
    s, p = mk(3, dx=5.0)
    b, d = r.episode_bonuses(0, s, p, [[0, 0]] * 3)
    assert np.all(b == 1.0) and d["clipped_steps"] == 3

    # 10. History cap evicts the OLDEST key, not the newest.
    r = StallRescue(2, history_cap=3)
    s = ((0.0, 0.0), 0.0, 1.0)
    path = [((10.0 * (i + 1), 0.0), 0.0, 0.0, 1.0) for i in range(5)]
    r.episode_bonuses(0, s, path, [[0, 0]] * 5)
    keys = list(r.action_history[0].keys())
    # pre-step states are x = 0, 10, 20, 30, 40 -> cells 0, 5, 10, 15, 20. Keep the newest three.
    assert [k[0] for k in keys] == [10, 15, 20], keys

    # 11. Same state discretisation for every bee.
    assert r.state_key((3.9, -0.1), 0.1, 12.0) == r.state_key((3.1, -1.9), 0.2, 14.9)

    print("stall_rescue: all 11 checks passed")
