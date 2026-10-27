import time
import json
import os 
from typing import Dict, Any, List 
from collections import deque

class BeeInstrumentation: 
    def __init__(self, log_dir:str = "outputs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.latencies: List[float] = []
        self.cell_read_counts: List[int] = []
        self.cell_hit_counts: List[int] = []
        self.steering_deviations: List[float] = []
        
        #episode trackers
        self.recent_rewards = deque(maxlen=20)
        self.recent_successes = deque(maxlen=10)
        
    def start_timer(self) -> float:
        return time.perf_counter()

    def record_step(
        self,
        start_time: float,
        probed_cells: int,
        hit_cells: int,
        chosen_steer: float,
        consensus_steer_log: float
    ):
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.latencies.append(elapsed_ms)
        self.cell_read_counts.append(probed_cells)
        self.cell_hit_counts.append(hit_cells)
        
        if hit_cells > 0 and consensus_steer_log is not None:
            dev = abs(chosen_steer - consensus_steer_log)
            self.steering_deviations.append(dev)

    def record_episode_end(self, reward: float, arrived_dest: bool, total_steps: int):
        self.recent_rewards.append(reward)
        self.recent_successes.append(1 if arrived_dest else 0)

    def get_summary(self) -> Dict[str, Any]:
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        avg_hits = sum(self.cell_hit_counts) / len(self.cell_hit_counts) if self.cell_hit_counts else 0.0
        avg_dev = sum(self.steering_deviations) / len(self.steering_deviations) if self.steering_deviations else 0.0
        success_rate = (sum(self.recent_successes) / len(self.recent_successes)) if self.recent_successes else 0.0
        
        return {
            'avg_latency_ms': round(avg_latency, 3),
            'avg_cell_hits': round(avg_hits, 2),
            'avg_steering_deviation': round(avg_dev, 4),
            'recent_success_rate': round(success_rate, 2),
            'episodes_tracked': len(self.recent_rewards)
        }

    def save_summary_json(self, filename: str = "phase_metrics.json"):
        summary = self.get_summary()
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  [Instrumentation] Telemetry saved to {filepath}")


if __name__ == "__main__":
    print("= Testing BeeInstrumentation =")
    inst = BeeInstrumentation(log_dir="/workspace/scratch/test_logs")
    
    #simulate steps
    t0 = inst.start_timer()
    time.sleep(0.002) 
    inst.record_step(t0, probed_cells=3, hit_cells=1, chosen_steer=0.1, consensus_steer_log=0.15)
    
    inst.record_episode_end(reward=15.0, arrived_dest=True, total_steps=50)
    
    summary = inst.get_summary()
    print("Summary:", summary)
    assert summary['recent_success_rate'] == 1.0
    
    inst.save_summary_json()
    print("-> BeeInstrumentation verification PASSED!")

