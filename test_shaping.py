"""
test_shaping.py

Tests for the pure shaping functions in shaping.py. Each test is a claim from
Documentation/shaping_spec.md.
"""

import math
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from shaping import potential, shaping_term, shaped_reward


def test_potential_definition():
    # Phi = beta * (S - lam * W)
    assert potential(0.0, 0.0, beta=2.0, lam=1.0) == 0.0, "empty table -> Phi 0"
    assert abs(potential(0.8, 0.0, beta=2.0, lam=1.0) - 1.6) < 1e-12
    assert abs(potential(0.0, 0.5, beta=2.0, lam=1.0) - (-1.0)) < 1e-12, "warning lowers Phi"
    assert abs(potential(0.8, 0.5, beta=2.0, lam=0.5) - (2.0 * (0.8 - 0.25))) < 1e-12
    print("  potential definition: PASS")


def test_potential_rejects_negative_weights():
    for beta, lam in ((-1.0, 1.0), (1.0, -0.5)):
        try:
            potential(0.5, 0.5, beta=beta, lam=lam)
            raise AssertionError(f"beta={beta}, lam={lam} should be rejected")
        except ValueError:
            pass
    print("  negative weights rejected: PASS")


def test_terminal_potential_is_zero():
    # Real terminal: Phi(s') = 0, so F = -Phi(s) whatever Phi(s') would have been.
    for phi_next in (-5.0, 0.0, 7.0):
        f = shaping_term(phi_s=1.5, phi_next=phi_next, gamma=0.99, next_is_terminal=True)
        assert abs(f - (-1.5)) < 1e-12, f"terminal F should be -Phi(s), got {f}"
    print("  terminal Phi = 0: PASS")


def test_timeout_uses_actual_potential():
    # Horizon truncation is not absorbing: F = gamma * Phi(s') - Phi(s).
    f = shaping_term(phi_s=1.0, phi_next=2.0, gamma=0.5, next_is_terminal=False)
    assert abs(f - (0.5 * 2.0 - 1.0)) < 1e-12, f
    print("  timeout uses actual Phi(s'): PASS")


def test_zero_table_gives_zero_shaping():
    f = shaping_term(0.0, 0.0, gamma=0.99, next_is_terminal=False)
    assert f == 0.0
    print("  zero table -> F = 0: PASS")


def test_shaped_reward_adds_F():
    r = shaped_reward(env_reward=3.0, phi_s=1.0, phi_next=2.0, gamma=0.5, next_is_terminal=False)
    assert abs(r - (3.0 + (0.5 * 2.0 - 1.0))) < 1e-12, r
    print("  r' = r + F: PASS")


def test_return_difference_is_boundary_only():
    # Key identity: over any trajectory of T steps with a fixed Phi,
    #   sum_t gamma^t F_t = gamma^T * Phi(s_T) - Phi(s_0)
    # (Phi(s_T) = 0 if the end is a real terminal). So the shaped return
    # differs from the raw return only by a function of the endpoints. Circling
    # a loop (s_T = s_0) therefore gains nothing: the difference is
    # (gamma^T - 1) * Phi(s_0), which is 0 only for gamma = 1 or Phi(s_0) = 0.
    rng = random.Random(0)
    gamma = 0.99
    for trial in range(200):
        T = rng.randint(1, 60)
        phis = [rng.uniform(-2.0, 2.0) for _ in range(T + 1)]
        end_terminal = rng.random() < 0.5
        if end_terminal:
            phis[-1] = 0.0  # real terminal: Phi(s_T) = 0 by definition
        raw = [rng.uniform(-1.0, 1.0) for _ in range(T)]

        shaped_return_gain = 0.0
        for t in range(T):
            last = t == T - 1
            f = shaping_term(phis[t], phis[t + 1], gamma, next_is_terminal=(last and end_terminal))
            shaped_return_gain += (gamma ** t) * f
        expected = (gamma ** T) * phis[-1] - phis[0]
        assert abs(shaped_return_gain - expected) < 1e-9, (trial, shaped_return_gain, expected)
    print("  return difference = gamma^T Phi(s_T) - Phi(s_0) over 200 random trajectories: PASS")


def test_loop_gains_nothing_when_gamma_is_one():
    # With gamma = 1 a closed loop (s_T = s_0) has zero total shaping: no circling bonus.
    phis = [0.7, 1.9, -0.3, 0.7]  # loop back to the start
    total = sum(shaping_term(phis[t], phis[t + 1], gamma=1.0, next_is_terminal=False) for t in range(3))
    assert abs(total) < 1e-12, total
    print("  closed loop, gamma=1 -> total shaping 0: PASS")


def test_gamma_out_of_range_rejected():
    for g in (-0.1, 1.1):
        try:
            shaping_term(0.0, 0.0, gamma=g, next_is_terminal=False)
            raise AssertionError(f"gamma={g} should be rejected")
        except ValueError:
            pass
    print("  gamma range check: PASS")


if __name__ == "__main__":
    print("=== shaping.py tests ===")
    test_potential_definition()
    test_potential_rejects_negative_weights()
    test_terminal_potential_is_zero()
    test_timeout_uses_actual_potential()
    test_zero_table_gives_zero_shaping()
    test_shaped_reward_adds_F()
    test_return_difference_is_boundary_only()
    test_loop_gains_nothing_when_gamma_is_one()
    test_gamma_out_of_range_rejected()
    print("ALL SHAPING TESTS PASSED")
