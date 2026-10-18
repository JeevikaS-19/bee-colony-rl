import math


def decay_rate_from_half_life(half_life_steps):
    """
    Per-training-step decay rate so a trace halves after `half_life_steps`.
    None or 0 means no decay.
    """
    if not half_life_steps:
        return 0.0
    return math.log(2) / half_life_steps


class PheromoneTable:
    def __init__(self, grid_size=2.0):
        self.grid_size = grid_size
        # Core Table Storage:
        # Key: (grid_x, grid_y) tuple
        # Value: A dictionary containing parallel, decoupled channels:
        # {
        #     "success": {"vigor": float, "timestamp": float, "meta": dict|None},
        #     "warning": {"vigor": float, "timestamp": float, "meta": dict|None}
        # }
        self.table = {}

    def _to_grid_key(self, location):
        x, y = location[0], location[1]

        grid_x = int(math.floor(x / self.grid_size))
        grid_y = int(math.floor(y / self.grid_size))

        return (grid_x, grid_y)

    def write(self, location, trace_type, vigor, step, meta=None, speed=None, steer=None):
        # speed (km/h) and steer (-1..1) describe how the bee passed this cell.
        # Location alone can't say how to pass: the crash spot also carries a
        # success trace. They are replaced together with vigor/meta.
        # `step` is the training step the trace was written at. Decay counts
        # steps, not seconds, so runs at different speeds decay the same way.
        if trace_type not in ["success", "warning"]:
            raise ValueError("trace_type must be either 'success' or 'warning'")
        # read_nearby skips zero-vigor cells, so a 0.0 write would be silently
        # invisible. Callers that have no vigor (NOVICE bees) must not write.
        if vigor <= 0.0:
            raise ValueError(f"vigor must be > 0 to be written, got {vigor}")
        key = self._to_grid_key(location)

        if key not in self.table:
            self.table[key] = {
                "success": {"vigor": 0.0, "timestamp": 0.0, "meta": None, "speed": None, "steer": None},
                "warning": {"vigor": 0.0, "timestamp": 0.0, "meta": None, "speed": None, "steer": None}
            }

        channel = self.table[key][trace_type]

        if vigor >= channel["vigor"]:
            old_vigor = channel["vigor"]
            channel["vigor"] = vigor
            channel["timestamp"] = step
            channel["meta"] = meta
            channel["speed"] = speed
            channel["steer"] = steer

            worker_id = meta.get("worker_id", "Unknown") if meta else "Unknown"
            trigger = meta.get("trigger", "unspecified") if meta else "unspecified"
            print(f"  [Pheromone Table] Grid {key} | Channel: {trace_type.upper()} | "
                  f"Updated by {worker_id} (vigor: {vigor:.2f}) | Trigger: {trigger} "
                  f"| Vigor shift: {old_vigor:.2f} -> {vigor:.2f}")
            return True
        else:
            return False

    def read_nearby(self, location, step, decay_rate, sensor_radius=10.0, sigma=6.0):
        # decay_rate is per training step (see decay_rate_from_half_life).
        # Required on purpose: a silent default is how the old 0.002 leaked in.
        ego_x, ego_y = location[0], location[1]

        sensed = {"success": 0.0, "warning": 0.0}

        # 1. broad phase grid cell search
        grid_search_range = int(math.ceil(sensor_radius / self.grid_size))
        center_key = self._to_grid_key(location)

        for dx in range(-grid_search_range, grid_search_range + 1):
            for dy in range(-grid_search_range, grid_search_range + 1):
                key = (center_key[0] + dx, center_key[1] + dy)
                if key not in self.table:
                    continue

                cell_center_x = (key[0] + 0.5) * self.grid_size
                cell_center_y = (key[1] + 0.5) * self.grid_size

                dist = math.sqrt((ego_x - cell_center_x) ** 2 + (ego_y - cell_center_y) ** 2)
                if dist > sensor_radius:
                    continue

                spatial_weight = math.exp(-(dist ** 2) / (2 * (sigma ** 2)))

                for trace_type in ["success", "warning"]:
                    channel = self.table[key][trace_type]
                    raw_vigor = channel["vigor"]

                    if raw_vigor == 0.0:
                        continue

                    delta_t = max(0.0, step - channel["timestamp"])
                    temporal_decay = math.exp(-decay_rate * delta_t)

                    effective_intensity = raw_vigor * spatial_weight * temporal_decay
                    sensed[trace_type] = max(sensed[trace_type], effective_intensity)

        return sensed

    def summarize(self, min_vigor=0.0):
        """
        Returns a list of plain-English lines describing every non-empty
        cell in the table — a quick human-readable sanity check without
        needing the render pipeline.

        Example line:
          "Turning guide near (9.0, 8.0) -- vigor 1.00, left by sandbox_bee
           after a 'arrive_dest' event"
        """
        entries = []  # (vigor, line) pairs, so we can sort before formatting

        for key, channels in self.table.items():
            cell_x = (key[0] + 0.5) * self.grid_size
            cell_y = (key[1] + 0.5) * self.grid_size

            for trace_type in ["success", "warning"]:
                ch = channels[trace_type]
                if ch["vigor"] <= min_vigor:
                    continue

                meta = ch["meta"] or {}
                worker = meta.get("worker_id", "unknown")
                trigger = meta.get("trigger", "unknown")
                label = "Turning guide" if trace_type == "success" else "Warning"

                line = (f"{label} near ({cell_x:.1f}, {cell_y:.1f}) -- "
                        f"vigor {ch['vigor']:.2f}, left by {worker} "
                        f"after a '{trigger}' event")
                entries.append((ch["vigor"], line))

        # Strongest traces first
        entries.sort(key=lambda pair: pair[0], reverse=True)
        return [line for _, line in entries]

    def print_summary(self, min_vigor=0.0):
        """Convenience wrapper: prints summarize() output directly."""
        lines = self.summarize(min_vigor=min_vigor)
        if not lines:
            print("  (pheromone table is empty)")
            return
        for line in lines:
            print(" ", line)