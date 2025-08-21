from dataclasses import dataclass

@dataclass
class CaseConfig:
    phi0: float = 1.0
    m2_L2: float = -2.5
    lambda_L2: float = 0.05
    bc: str = "dirichlet"
    f_double_trace: float = 0.0

@dataclass
class TrainConfig:
    r_min: float = 0.01
    r_max: float = 8.0
    n_points: int = 200
    max_epochs: int = 800
    lr: float = 1e-2
    step_size: int = 300
    gamma: float = 0.9
    patience: int = 200
    success_loss: float = 0.5
    target_loss: float = 1e-2
    good_enough_loss: float = 1e-1
    use_lbfgs: bool = True
    lbfgs_steps: int = 60
    verbose: bool = True
    matplotlib_backend_agg: bool = True
