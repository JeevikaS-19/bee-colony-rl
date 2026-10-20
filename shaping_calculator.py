import math
from typing import Dict, Any, Tuple
from config import BeeConfig, default_config 

class ShapingCalculator:
    def __init__(self, config: BeeConfig = default_config):
        self.config = config
        self.eta_success = config.eta_success
        self.eta_crash = config.eta_crash
        #Note: reading enabled versus disabled, phase 2 write-only vs phase 3 read/write. 
        self.reading_enabled = False

    def set_reading_enabled(self,enabled: bool):
        self.reading_enabled = enabled
    def compute_potential(
            self,
            agent_heading: float,
            inferred_steer: float,
            agent_speed: float,
            corridor_readout: Dict[str, Any]
    ) -> float:
        if not self.reading_enabled:
            return 0.0
        raw_success = corridor_readout.get('raw_success_dances', [])
        raw_crash = corridor_readout.get('raw_crash_dances', [])
        phi_success = 0.0
        if raw_success:
            for dist_m, weight, dance in raw_success:
                dance_steer = dance.get('steer', 0.0)
                steer_diff = abs(inferred_steer - dance_steer)
                alignment = math.cos(steer_diff*(math.pi/2.0))

                phi_success += weight * alignment

        phi_crash = 0.0
        if raw_crash:
            for dist_m, weight, dance in raw_crash:
                crash_type = dance.get('crash_type', 'general')
                proximity = math.exp(-dist_m/3.0)

                if crash_type in ['vehicle', 'obstacle', 'wall']:
                    speed_factor = max(0.1, agent_speed/15.0)
                    penalty = proximity*speed_factor
                elif crash_type in ['curb', 'offroad', 'out_of_road']:
                    hazard_steer = dance.get('steer', 0.0)
                    steer_towards = 1.0 - (abs(inferred_steer-hazard_steer)/2.0)
                    penalty = proximity*max(0.2, steer_towards)
                else:
                    penalty = proximity
                phi_crash += weight*penalty
        phi_total = (self.eta_success*phi_success) - (self.eta_crash*phi_crash)
        return phi_total

    def compute_shaped_reward(
            self,
            base_reward:float,
            phi_current: float,
            phi_next: float,
            gamma: float = 0.99
    )-> float:

        shaping = (gamma*phi_next) - phi_current
        return base_reward + shaping 

if __name__ == "__main__":
    print("= Testing ShapingCalculator with Reading Flag & Crash Types =")
    calc = ShapingCalculator()
    
    mock_readout = {
        'raw_success_dances': [
            (2.0, 0.9, {'steer': 0.2, 'vigor': 0.8}),
            (4.0, 0.7, {'steer': 0.3, 'vigor': 0.8})
        ],
        'raw_crash_dances': [
            (3.0, 0.8, {'steer': -0.8, 'vigor': 0.9, 'crash_type': 'curb'})
        ]
    }
    
    #1. Reading disabled -> phi = 0.0
    calc.set_reading_enabled(False)
    phi_disabled = calc.compute_potential(0.0, 0.2, 10.0, mock_readout)
    assert phi_disabled == 0.0, "Disabled reader must return 0.0 potential"
    print("-> Reading disabled check passed!")

    #2. Reading enabled
    calc.set_reading_enabled(True)
    phi_aligned = calc.compute_potential(0.0, 0.2, 10.0, mock_readout)
    phi_towards_curb = calc.compute_potential(0.0, -0.8, 10.0, mock_readout)
    
    print(f"Phi Aligned (steer=0.2): {phi_aligned:.4f}")
    print(f"Phi Towards Curb (steer=-0.8): {phi_towards_curb:.4f}")
    assert phi_aligned > phi_towards_curb, "Steering towards curb hazard must have lower potential"
    
    print("-> ShapingCalculator updated verification PASSED!")
