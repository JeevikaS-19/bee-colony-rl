import math
from typing import Dict, Tuple, Any, Optional

from config import default_config

# MetaDrive terminal keys that count as a crash. Anything else is rejected on write.
CRASH_TYPE_TO_CHANNEL = {
    "crash_vehicle": "crash_vehicle_dances",
    "out_of_road": "crash_offroad_dances",
    "crash_object": "crash_offroad_dances",
    "crash_building": "crash_offroad_dances",
}


def decay_rate_from_half_life(half_life_steps: Optional[float]) -> float:
    """Per-step decay rate. None or 0 means no decay."""
    if not half_life_steps:
        return 0.0
    return math.log(2.0) / half_life_steps


def decayed_vigor(dance: Dict[str, Any], step_now: int, decay_rate: float) -> float:
    """vigor * exp(-rate * (step_now - step_written)). Going back in time is an error."""
    age = step_now - dance['timestamp']
    if age < 0:
        raise ValueError(f"step_now ({step_now}) is before the write step ({dance['timestamp']})")
    return dance['vigor'] * math.exp(-decay_rate * age)


class PheromoneTableV2:
    def __init__(self, grid_size: float = default_config.grid_size,
                 heading_bins: int = default_config.heading_bins):
        self.grid_size = grid_size
        self.heading_bins = heading_bins
        self.table: Dict[Tuple[int, int], Dict[str, Dict[int, Dict[str, Any]]]] = {}

    def _to_grid_key(self, location: Tuple[float, float]) -> Tuple[int, int]:
        x, y = location[0], location[1]
        grid_x = int(math.floor(x / self.grid_size))
        grid_y = int(math.floor(y / self.grid_size))
        return (grid_x, grid_y)

    def _to_heading_bin(self, heading_rad: float) -> int:
        two_pi = 2.0 * math.pi
        normalized_angle = heading_rad % two_pi
        bin_width = two_pi / self.heading_bins
        return int(math.floor(normalized_angle / bin_width)) % self.heading_bins

    def _circular_bin_distance(self, a: int, b: int) -> int:
        diff = abs(a - b)
        return min(diff, self.heading_bins - diff)

    def write_dance(
            self,
            location: Tuple[float, float],
            heading: float,
            channel: str,
            steer: float,
            vigor: float,
            step: int,
            decay_rate: float,
            speed: float = 0.0,
            progress_after: float = 0.0,
            crash_type: str = "none",
            meta: Optional[Dict[str, Any]] = None
    ) -> bool:
        """channel is 'success' or 'crash'. A crash needs a MetaDrive crash_type."""
        if channel == "success":
            target_channel = 'success_dances'
        elif channel == "crash":
            if crash_type not in CRASH_TYPE_TO_CHANNEL:
                raise ValueError(
                    f"crash_type must be one of {sorted(CRASH_TYPE_TO_CHANNEL)}, got {crash_type!r}")
            target_channel = CRASH_TYPE_TO_CHANNEL[crash_type]
        else:
            raise ValueError("Channel must be 'success' or 'crash'")

        if vigor <= 0:
            return False

        key = self._to_grid_key(location)
        h_bin = self._to_heading_bin(heading)

        if key not in self.table:
            self.table[key] = {
                'success_dances': {},
                'crash_vehicle_dances': {},
                'crash_offroad_dances': {},
            }

        channel_dict = self.table[key][target_channel]
        existing = channel_dict.get(h_bin)
        # Compare against the decayed strength of what is already there.
        old_vigor = decayed_vigor(existing, step, decay_rate) if existing else 0.0

        if vigor >= old_vigor:
            channel_dict[h_bin] = {
                'heading': heading,
                'heading_bin': h_bin,
                'steer': steer,
                'speed': speed,
                'vigor': vigor,
                'timestamp': step,  # training step, never wall-clock
                'progress_after': progress_after,
                'crash_type': crash_type if channel == "crash" else "none",
                'meta': meta or {}
            }
            return True

        return False

    def get_lookahead_keys(
            self,
            position: Tuple[float, float],
            heading_angle: float,
            num_cells: int = default_config.lookahead_cells
    ):
        lookahead = []
        x, y = position[0], position[1]

        for i in range(1, num_cells + 1):
            dist_meters = i * self.grid_size
            dx = dist_meters * math.cos(heading_angle)
            dy = dist_meters * math.sin(heading_angle)
            cell_key = self._to_grid_key((x + dx, y + dy))
            lookahead.append((cell_key, dist_meters))
        return lookahead

    def read_cell_dances(
            self,
            key: Tuple[int, int],
            heading_angle: float
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Returns (success, crash_vehicle, crash_offroad) for one cell, each the best dance
        for this heading or the nearest heading bin. Dances are undecayed; apply
        decayed_vigor() with the current step before using their vigor.
        """
        if key not in self.table:
            return None, None, None

        h_bin = self._to_heading_bin(heading_angle)
        cell_data = self.table[key]
        picked = []
        for name in ('success_dances', 'crash_vehicle_dances', 'crash_offroad_dances'):
            channel_dict = cell_data[name]
            best = channel_dict.get(h_bin)
            if best is None and channel_dict:
                best = min(
                    channel_dict.values(),
                    key=lambda d: self._circular_bin_distance(d['heading_bin'], h_bin)
                )
            picked.append(best)
        return tuple(picked)

    def __len__(self) -> int:
        return len(self.table)
