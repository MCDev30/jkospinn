from pkg import *

class Config(NamedTuple):
    """Global configuration for experiments"""
    hidden_dims: Tuple[int, ...] = (10, 10, 10)
    activation: str = "swish"
    fourier_features: int = 32
    fourier_scale: float = 3.0

    n_epochs: int = 5000
    batch_size_data: int = 32
    batch_size_colloc: int = 64
    lr_network: float = 1e-2
    lr_physics: float = 1e-2
    lr_schedule_decay: float = 0.995

    lambda_data: float = 2.0
    lambda_physics: float = 1.0
    lambda_guidance: float = 10.0

    # DSM: bruit standard
    gamma_noise: float = 0.2

    n_hutchinson_samples: int = 4

    # Génération données
    dt_sim: float = 0.05
    dt_obs: float = 0.2
    T: float = 2.0
    n_trajectories: int = 10

    x_min: float = -3.0
    x_max: float = 3.0

    seed: int = 42
