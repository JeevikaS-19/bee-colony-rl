import math
from typing import Dict, Any

from config import BeeConfig, default_config

# Speed (km/h) at which a vehicle-crash penalty reaches full strength. Tune later.
CRASH_SPEED_REF_KMH = 40.0


class ShapingCalculator:
    """
    Potential Phi(s) for reward-side shaping. Phi depends only on the state
    (position, heading, speed) and the decayed corridor read at that state.
    It never takes the policy's action, so F = gamma*Phi(s') - Phi(s) is a
    valid potential-based term.
    """
    def __init__(self, config: BeeConfig = default_config):
        self.config = config
        self.eta_success = config.eta_success
        self.eta_crash = config.eta_crash
        self.reading_enabled = False

    def set_reading_enabled(self, enabled: bool):
        self.reading_enabled = enabled

    def compute_potential(
            self,
            agent_heading: float,
            agent_speed: float,
            corridor_readout: Dict[str, Any]
    ) -> float:
        if not self.reading_enabled:
            return 0.0

        # Success: reward heading the same way as the success traces around you.
        # Both headings are states, so no action enters here.
        phi_success = 0.0
        for dist_m, weight, dance in corridor_readout['raw_success_dances']:
            phi_success += weight * math.cos(agent_heading - dance['heading'])

        # Crash: penalise the state that led to a crash.
        phi_crash = 0.0
        for dist_m, weight, dance in corridor_readout['raw_crash_dances']:
            proximity = math.exp(-dist_m / 3.0)
            if dance.get('crash_type') == 'crash_vehicle':
                # Vehicle crash: going fast here is the hazard.
                speed_factor = max(0.1, min(1.0, agent_speed / CRASH_SPEED_REF_KMH))
                penalty = proximity * speed_factor
            else:
                # Off-road, object, or building: heading the way the crash went is the hazard.
                alignment = max(0.0, math.cos(agent_heading - dance['heading']))
                penalty = proximity * max(0.2, alignment)
            phi_crash += weight * penalty

        return (self.eta_success * phi_success) - (self.eta_crash * phi_crash)

    def compute_shaped_reward(
            self,
            base_reward: float,
            phi_current: float,
            phi_next: float,
            gamma: float = 0.99
    ) -> float:
        """r + F with F = gamma*Phi(s') - Phi(s). No event bonus. Pass phi_next = 0 at real terminals."""
        return base_reward + (gamma * phi_next - phi_current)


if __name__ == "__main__":
    print("= Testing state-only ShapingCalculator =")
    calc = ShapingCalculator()

    mock_readout = {
        'raw_success_dances': [
            (2.0, 0.9, {'heading': 0.0, 'steer': 0.2, 'vigor_now': 0.8}),
            (4.0, 0.7, {'heading': 0.0, 'steer': 0.3, 'vigor_now': 0.8}),
        ],
        'raw_crash_dances': [
            (3.0, 0.8, {'heading': math.pi, 'crash_type': 'out_of_road', 'vigor_now': 0.9}),
        ],
    }

    # 1. Reading disabled -> Phi = 0, so F = 0 and the reward is unchanged.
    calc.set_reading_enabled(False)
    assert calc.compute_potential(0.0, 10.0, mock_readout) == 0.0
    print("-> Reading disabled check passed!")

    # 2. Facing the success traces scores higher than facing away from them.
    calc.set_reading_enabled(True)
    phi_along = calc.compute_potential(0.0, 10.0, mock_readout)
    phi_away = calc.compute_potential(math.pi, 10.0, mock_readout)
    print(f"Phi along traces: {phi_along:.4f} | Phi facing away: {phi_away:.4f}")
    assert phi_along > phi_away

    # 3. F with no event bonus: terminal next state (phi_next = 0).
    f = calc.compute_shaped_reward(1.0, phi_along, 0.0, gamma=0.99) - 1.0
    assert abs(f - (-phi_along)) < 1e-9
    print("-> State-only shaping verification PASSED!")
