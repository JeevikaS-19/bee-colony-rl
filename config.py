import os 
from dataclasses import dataclass, field
@dataclass
class BeeConfig:
    #Environment and physical sim settings 
    map_type: str = "C"
    use_render: bool = False
    horizon: int = 500
    obs_dim: int = 259
    action_dim: int = 2

    #sac agent and neural network hyperparameters 
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-5
    alpha_init: float = 1.0
    alpha_floor: float = 0.03
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_capacity: int = 200000
    warmup_steps: int = 1000

    #pheromone table and waggle dance 
    grid_size: float = 2.0
    sensor_radius: float = 10.0
    lookahead_cells: int = 3
    heading_bins: int = 8
    gaussian_sigma: float = 6.0
    half_life_frac: float = 0.15  # fraction of total training steps; 0 = no decay (same as sandbox_bee2_training.py)

    #stigmergic reward shaping weights 
    eta_success: float = 0.50
    eta_crash: float = 1.00

    #phase execution and gating 
    phase1_total_steps: int = 20000
    phase1_target_success_rate: float = 0.30
    phase2_target_dance_entries: int = 75
    phase3_gate_eval_episodes: int = 10
    phase3_gate_min_successes: int = 3
    phase4_elder_count: int =1
    phase4_learner_count:int = 3

    output_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "outputs"))
    checkpoint_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "checkpoints"))

    def get_phase_dir(self, phase_name:str)->str:
        path = os.path.join(self.output_dir, phase_name)
        os.makedirs(path,exist_ok=True)
        return path
default_config = BeeConfig()
