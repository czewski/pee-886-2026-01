import argparse
from pathlib import Path

import numpy as np
import pennylane as qml
from matplotlib import pyplot as plt


def build_hybrid_classifier_qnode(nqubits: int, nlayers: int):
    dev = qml.device("default.qubit", wires=nqubits)

    @qml.qnode(dev)
    def qnn_layer(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="X")
        for layer_index in range(nlayers):
            for i in range(nqubits):
                qml.RY(weights[layer_index, i], wires=i)
            for i in range(nqubits):
                j = (i + 1) % nqubits
                qml.CNOT(wires=(i, j))
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(nqubits))

    return qnn_layer


def build_hybrid_model_qnode(nqubits: int, nlayers: int):
    dev = qml.device("default.qubit", wires=nqubits)

    @qml.qnode(dev)
    def quantum_circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(nqubits), normalize=True)
        qml.BasicEntanglerLayers(weights, wires=range(nqubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(nqubits)]

    return quantum_circuit


def build_simple_cnn_qml_qnode(nqubits: int, nlayers: int, qml_ops: str):
    dev = qml.device("default.qubit", wires=nqubits)

    if qml_ops == "baseline":

        @qml.qnode(dev)
        def qnn_layer(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="Y")
            for layer_idx in range(nlayers):
                for wire in range(nqubits):
                    qml.RY(weights[layer_idx, wire], wires=wire)
                for wire in range(nqubits):
                    qml.CNOT(wires=(wire, (wire + 1) % nqubits))
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(nqubits))

        weights = np.full((nlayers, nqubits), 0.2)
        return qnn_layer, weights

    @qml.qnode(dev)
    def qnn_layer(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="X")
        for layer_idx in range(nlayers):
            for wire in range(nqubits):
                qml.RX(weights[layer_idx, wire, 0], wires=wire)
                qml.RZ(weights[layer_idx, wire, 1], wires=wire)
            for wire in range(nqubits):
                qml.CZ(wires=(wire, (wire + 1) % nqubits))
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(nqubits))

    weights = np.full((nlayers, nqubits, 2), 0.2)
    return qnn_layer, weights


def draw_circuit(name: str, circuit, inputs, weights, save_path: Path):
    print(f"\n{name}")
    print("-" * len(name))
    print(qml.draw(circuit)(inputs, weights))
    fig, _ = qml.draw_mpl(circuit)(inputs, weights)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw quantum parts used by group_03 models."
    )
    parser.add_argument(
        "--model",
        choices=["hybrid_classifier", "hybrid_model", "simple_cnn_qml", "all"],
        default="all",
    )
    parser.add_argument(
        "--qml-ops",
        choices=["baseline", "alt_ops"],
        default="baseline",
        help="QNN ops variant used by simple_cnn_qml.",
    )
    parser.add_argument("--nqubits", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Experiment name. If set, images are saved to outputs/<exp-name>/",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for saved images.",
    )
    args = parser.parse_args()
    if args.outdir:
        outdir = Path(args.outdir)
    elif args.exp_name:
        outdir = Path("outputs") / args.exp_name
    else:
        outdir = Path(__file__).parent
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.exp_name}_" if args.exp_name else ""

    if args.model in ("hybrid_classifier", "all"):
        qnode = build_hybrid_classifier_qnode(args.nqubits, args.nlayers)
        inputs = np.linspace(0.1, 0.4, args.nqubits)
        weights = np.full((args.nlayers, args.nqubits), 0.2)
        save_path = outdir / f"{prefix}hybrid_classifier_qnn_layer.png"
        draw_circuit(
            "HybridClassifier quantum part (qnn_layer)",
            qnode,
            inputs,
            weights,
            save_path,
        )

    if args.model in ("hybrid_model", "all"):
        qnode = build_hybrid_model_qnode(args.nqubits, args.nlayers)
        inputs = np.linspace(0.01, 0.16, 2**args.nqubits)
        weights = np.full((args.nlayers, args.nqubits), 0.3)
        save_path = outdir / f"{prefix}hybrid_model_quantum_circuit.png"
        draw_circuit(
            "HybridModel quantum part (quantum_circuit)",
            qnode,
            inputs,
            weights,
            save_path,
        )

    if args.model in ("simple_cnn_qml", "all"):
        qnode, weights = build_simple_cnn_qml_qnode(
            args.nqubits, args.nlayers, args.qml_ops
        )
        inputs = np.linspace(0.1, 0.4, args.nqubits)
        save_path = outdir / f"{prefix}simple_cnn_qml_{args.qml_ops}_qnn_layer.png"
        draw_circuit(
            f"SimpleCNNQML quantum part ({args.qml_ops})",
            qnode,
            inputs,
            weights,
            save_path,
        )


if __name__ == "__main__":
    main()
