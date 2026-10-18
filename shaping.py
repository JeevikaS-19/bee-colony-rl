"""
Potential-based reward shaping from pheromone traces.
Definitions are in Documentation/shaping_spec.md (sections 2-3). This module
is pure: it takes numbers in and returns numbers out. Reading the table is
the caller's job.

  Phi(s)        = beta * (S(x) - lam * W(x))      S, W = success/warning read at x
  Phi(terminal) = 0                                 real terminals only
  F(s, s')      = gamma * Phi(s') - Phi(s)
  r'            = r + F
"""


def potential(success, warning, beta, lam):
    """Phi of one state, from the success and warning reads at its position."""
    if beta < 0.0 or lam < 0.0:
        raise ValueError(f"beta and lam must be >= 0, got beta={beta}, lam={lam}")
    return beta * (success - lam * warning)


def shaping_term(phi_s, phi_next, gamma, next_is_terminal):
    """
    F(s, s') = gamma * Phi(s') - Phi(s).
    next_is_terminal: True for a real terminal (arrival, crash, out-of-road),
    where Phi(s') = 0. False for a horizon timeout, which is a truncation,
    so the actual Phi at the truncated state is used.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    phi_next_eff = 0.0 if next_is_terminal else phi_next
    return gamma * phi_next_eff - phi_s


def shaped_reward(env_reward, phi_s, phi_next, gamma, next_is_terminal):
    """r' = r + F(s, s')."""
    return env_reward + shaping_term(phi_s, phi_next, gamma, next_is_terminal)
