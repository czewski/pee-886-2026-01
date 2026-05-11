import torch
import torch.nn as nn

__all__ = ["QmlHybridCnn"]


def _build_torch_layer(variant: str, n_qubits: int, n_q_layers: int):
    try:
        import pennylane as qml
        from pennylane import qnn
    except ImportError as exc:
        raise ImportError(
            "Pennylane is required for QML models. Install project requirements first."
        ) from exc

    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="torch")
    def quantum_circuit(inputs, weights):
        if variant == "qml_baseline":
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_q_layers):
                for wire in range(n_qubits):
                    qml.RY(weights[layer, wire], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=(wire, (wire + 1) % n_qubits))
        elif variant == "qml_strong":
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        elif variant == "qml_data_reupload":
            for layer in range(n_q_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                for wire in range(n_qubits):
                    qml.RZ(weights[layer, wire, 0], wires=wire)
                    qml.RY(weights[layer, wire, 1], wires=wire)
                for wire in range(n_qubits):
                    qml.CZ(wires=(wire, (wire + 1) % n_qubits))
        else:
            raise ValueError(f"Unsupported QML variant: {variant}")

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    if variant == "qml_baseline":
        weight_shapes = {"weights": (n_q_layers, n_qubits)}
    elif variant == "qml_strong":
        weight_shapes = {"weights": (n_q_layers, n_qubits, 3)}
    else:
        weight_shapes = {"weights": (n_q_layers, n_qubits, 2)}

    return qnn.TorchLayer(quantum_circuit, weight_shapes)


class QmlHybridCnn(nn.Module):
    def __init__(
        self,
        variant: str = "qml_baseline",
        n_qubits: int = 4,
        n_q_layers: int = 2,
        num_classes: int = 10,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_qubits),
            nn.Tanh(),
        )
        self.quantum = _build_torch_layer(variant, n_qubits, n_q_layers)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.quantum(x)
        return self.classifier(x)
