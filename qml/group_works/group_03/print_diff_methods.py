import pennylane as qml
import torch


def _hybrid_model_qnode(nqubits, nlayers):
    dev = qml.device("default.qubit", wires=nqubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(nqubits), normalize=True)
        qml.BasicEntanglerLayers(weights, wires=range(nqubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(nqubits)]

    return circuit


def _hybrid_flatten_qnode(nqubits, nlayers):
    dev = qml.device("default.qubit", wires=nqubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="X")
        for layer_idx in range(nlayers):
            for wire in range(nqubits):
                qml.RY(weights[layer_idx, wire], wires=wire)
            for wire in range(nqubits):
                qml.CNOT(wires=(wire, (wire + 1) % nqubits))
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(nqubits))

    return circuit


def _simple_cnn_qml_qnode(nqubits, nlayers, qml_ops):
    dev = qml.device("default.qubit", wires=nqubits)

    if qml_ops == "baseline":

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="Y")
            for layer_idx in range(nlayers):
                for wire in range(nqubits):
                    qml.RY(weights[layer_idx, wire], wires=wire)
                for wire in range(nqubits):
                    qml.CNOT(wires=(wire, (wire + 1) % nqubits))
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(nqubits))

        return circuit

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="X")
        for layer_idx in range(nlayers):
            for wire in range(nqubits):
                qml.RX(weights[layer_idx, wire, 0], wires=wire)
                qml.RZ(weights[layer_idx, wire, 1], wires=wire)
            for wire in range(nqubits):
                qml.CZ(wires=(wire, (wire + 1) % nqubits))
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(nqubits))

    return circuit


def _resolved_gradient_method(qnode, inputs, weights):
    qnode(inputs, weights)
    grad_fn = getattr(qnode, "gradient_fn", None)
    if isinstance(grad_fn, str):
        return grad_fn
    if grad_fn is None:
        return "None"
    if hasattr(grad_fn, "__name__"):
        return grad_fn.__name__
    return type(grad_fn).__name__


def main():
    experiments = [
        ("exp_1_hybrid_cnn_composta_gabriel", _hybrid_model_qnode(4, 2), torch.linspace(0.01, 0.16, 16), torch.full((2, 4), 0.3)),
        ("exp_2_hybrid_flatten_qml", _hybrid_flatten_qnode(4, 2), torch.linspace(0.1, 0.4, 4), torch.full((2, 4), 0.2)),
        ("exp_3_hybrid_qml_improved", _hybrid_flatten_qnode(10, 4), torch.linspace(0.1, 1.0, 10), torch.full((4, 10), 0.2)),
        ("exp_4_hybrid_cnn_simples_qml", _simple_cnn_qml_qnode(4, 2, "baseline"), torch.linspace(0.1, 0.4, 4), torch.full((2, 4), 0.2)),
        ("exp_5_hybrid_cnn_simples_qml_alt_ops", _simple_cnn_qml_qnode(4, 2, "alt_ops"), torch.linspace(0.1, 0.4, 4), torch.full((2, 4, 2), 0.2)),
        ("exp_6_hybrid_cnn_simples_qml_alt_ops_10q", _simple_cnn_qml_qnode(10, 2, "alt_ops"), torch.linspace(0.1, 1.0, 10), torch.full((2, 10, 2), 0.2)),
    ]

    print("Experiment diff_method report")
    print("============================")
    for name, qnode, inputs, weights in experiments:
        configured = qnode.diff_method
        resolved = _resolved_gradient_method(qnode, inputs, weights)
        print(f"{name}")
        print(f"  configured diff_method: {configured}")
        print(f"  resolved gradient method: {resolved}")

    print("exp_b_cnn_benchmark")
    print("  configured diff_method: N/A (no QNode)")
    print("  resolved gradient method: N/A (no QNode)")


if __name__ == "__main__":
    main()
