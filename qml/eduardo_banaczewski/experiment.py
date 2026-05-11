import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

__all__ = ["CifarExperimentConfig", "create_output_dir", "set_global_seed"]


@dataclass(frozen=True)
class CifarExperimentConfig:
    experiment_name: str = "cifar_experiment"
    model_name: str = "cnn_benchmark"
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 2
    n_folds: int = 5
    data_dir: str = "./data/eduardo_banaczewski"
    output_dir: str = "./outputs"
    n_qubits: int = 4
    n_q_layers: int = 2
    deterministic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize experiment configuration to a plain dictionary."""
        return asdict(self)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set global random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def create_output_dir(config: CifarExperimentConfig) -> Path:
    """Create and return the output directory for one experiment."""
    run_dir = Path(config.output_dir) / config.experiment_name
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
