import math
import time 
class PheromoneTable:
    def __init__(self, grid_size=2.0):
        self.grid_size = grid_size
        # Core Table Storage:
        # Key: (grid_x, grid_y) tuple
        # Value: A dictionary containing parallel, decoupled channels:
        # {
        #     "success": {"vigor": float, "timestamp": float},
        #     "warning": {"vigor": float, "timestamp": float}
        # }
        self.table = {}
    def _to_grid_key(self, location):
        x,y = location[0], location[1]

        grid_x = int(math.floor(x/self.grid_size))
        grid_y = int(math.floor(y/self.grid_size))

        return (grid_x, grid_y)

    def write(self, location, trace_type, vigor, meta = None):
        if trace_type not in ["success", "warning"]:
            raise ValueError("trace_type must be either 'success' or 'warning'")
        key = self._to_grid_key(location)
        current_time = time.time()

        if key not in self.table: 
            self.table[key] = {
                "success" : {"vigor": 0.0, "timestamp": 0.0, "meta": None},
                "warning": {"vigor" : 0.0, "timestamp": 0.0, "meta" : None}
            }

        channel = self.table[key][trace_type]

        if vigor >= channel["vigor"]:
            old_vigor = channel["vigor"]
            channel["vigor"] = vigor
            channel["timestamp"] = current_time
            channel["meta"] = meta

            worker_id = meta.get("worker_id", "Unknown") if meta else "Unknown"
            trigger = meta.get("trigger", "unspecified") if meta else "unspecified"
            print(f"  [Pheromone Table] Grid {key} | Channel: {trace_type.upper()} | "
                  f"Updated by {worker_id} (vigor: {vigor:.2f}) | Trigger: {trigger} "
                  f"| Vigor shift: {old_vigor:.2f} -> {vigor:.2f}")
            return True
        else:
            return False

    def read_nearby(self, location, sensro_radius=10.0, sigma = 6.0, decay_rate=0.002):
        ego_x, ego_y = location[0], location[1]
        current_time = time.time()

        sensed = {"success": 0.0, "warning": 0.0}

        #1. broad phase grid cell search
        grid_search_range = int(math.ceil(sensro_radius/self.grid_size)) 
        #fixed this error math.cell --> math.ceil
        center_key = self._to_grid_key(location)

        for dx in range(-grid_search_range, grid_search_range+1):
            for dy in range(-grid_search_range, grid_search_range+1):
                key = (center_key[0]+ dx, center_key[1]+dy)
                if key not in self.table:
                    continue

                cell_center_x = (key[0] + 0.5) * self.grid_size
                cell_center_y = (key[1] + 0.5) * self.grid_size

                dist = math.sqrt((ego_x - cell_center_x)**2 + (ego_y - cell_center_y)**2)
                if dist > sensro_radius:
                    continue
                #fixed indentation
                spatial_weight = math.exp(-(dist**2)/(2*(sigma**2)))

                for trace_type in ["success", "warning"]:
                    channel = self.table[key][trace_type]
                    raw_vigor = channel["vigor"]

                    if raw_vigor == 0.0:
                        continue

                    delta_t = max(0.0, current_time - channel["timestamp"])
                    temporal_decay = math.exp(-decay_rate * delta_t)

                    effective_intensity = raw_vigor * spatial_weight * temporal_decay
                    sensed[trace_type] = max(sensed[trace_type], effective_intensity)

        return sensed