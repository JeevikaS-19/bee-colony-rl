# TODO: broken until v2 is wired in. read_cell_dances() now returns a 3-tuple
# (success, crash_vehicle, crash_offroad), and read_corridor() must apply decayed_vigor().
import math
from typing import Dict, List, Tuple, Any, Optional
from pheromone_table_v2 import PheromoneTableV2

class DanceReader:
    def __init__(self, table: PheromoneTableV2, num_lookahead_cells: int = 3, sigma: float = 6.0):
        self.table = table
        self.num_cells = num_lookahead_cells
        self.sigma = sigma

    def read_corridor(
            self,
            position: Tuple[float, float],
            heading_angle: float
    ) -> Dict[str, Any]:

        lookahead_keys = self.table.get_lookahead_keys(
            position=position,
            heading_angle=heading_angle,
            num_cells=self.num_cells
        )

        corridor_cells = []
        raw_success = []
        raw_warning = []

        total_s_weight = 0.0
        weighted_steer_sum = 0.0

        for key, dist_m in lookahead_keys:
            w_spatial = math.exp(-(dist_m ** 2) / (2.0 * (self.sigma ** 2)))
            s_dance, w_dance = self.table.read_cell_dances(key, heading_angle)

            corridor_cells.append({
                'cell_key': key,
                'distance_m': dist_m,
                'spatial_weight': w_spatial,
                'success_dance': s_dance,
                'warning_dance': w_dance
            })

            if s_dance is not None:
                eff_w = w_spatial * s_dance['vigor']
                raw_success.append((dist_m, w_spatial, s_dance))
                total_s_weight += eff_w
                weighted_steer_sum += s_dance['steer'] * eff_w

            if w_dance is not None:
                raw_warning.append((dist_m, w_spatial, w_dance))

        consensus_steer_log = (weighted_steer_sum / total_s_weight) if total_s_weight > 0.0 else 0.0

        return {
            'corridor_cells': corridor_cells,
            'raw_success_dances': raw_success,
            'raw_warning_dances': raw_warning,
            'consensus_steer_log': consensus_steer_log
        }

if __name__ == "__main__":
    table = PheromoneTableV2()
    # Write a dummy success dance at world (6.0, 20.0) -> grid cell (3, 10)
    table.write_dance((6.0, 20.0), heading=0.0, channel="success", steer=0.25, vigor=0.9, step=0, speed=10.0, progress_after=4.5)

    reader = DanceReader(table, num_lookahead_cells=3)
    readout = reader.read_corridor((0.0, 20.0), heading_angle=0.0)

    print("Corridor Cells:", len(readout['corridor_cells']))
    print("Raw Success Dances:", len(readout['raw_success_dances']))
    print("Consensus Steer Log:", readout['consensus_steer_log'])
    assert len(readout['raw_success_dances']) == 1
    assert abs(readout['consensus_steer_log'] - 0.25) < 0.001
    print("\n-> DanceReader verification PASSED!")
