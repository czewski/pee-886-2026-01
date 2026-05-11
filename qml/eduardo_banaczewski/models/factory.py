import torch.nn as nn

from qml.eduardo_banaczewski.experiment import CifarExperimentConfig
from qml.eduardo_banaczewski.models.cnn_benchmark import CnnBenchmark
from qml.eduardo_banaczewski.models.qml_hybrid import QmlHybridCnn

__all__ = ["create_model"]


def create_model(config: CifarExperimentConfig, num_classes: int = 10) -> nn.Module:
    """Instantiate the selected model architecture from experiment config."""
    if config.model_name == "cnn_benchmark":
        return CnnBenchmark(num_classes=num_classes)
    if config.model_name in {"qml_baseline", "qml_strong", "qml_data_reupload"}:
        return QmlHybridCnn(
            variant=config.model_name,
            n_qubits=config.n_qubits,
            n_q_layers=config.n_q_layers,
            num_classes=num_classes,
        )
    raise ValueError(
        f"Unsupported model_name '{config.model_name}'. "
        "Expected one of: cnn_benchmark, qml_baseline, qml_strong, qml_data_reupload."
    )
