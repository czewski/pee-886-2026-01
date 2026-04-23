import argparse
import os
import random

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from qml.group_works.group_03.models.simple_cnn_qml import SimpleCNNQML
from qml.group_works.group_03.trainer import Trainer


def mnist_loader(batch_size, num_workers, pin_memory):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader, train_dataset, test_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nqubits", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--epochs-per-fold", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--exp-name", type=str, default="simple_cnn_qml")
    parser.add_argument(
        "--qml-ops",
        choices=["baseline", "alt_ops"],
        default="baseline",
    )
    parser.add_argument("--fit", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_qnn_layer(nqubits, nlayers, qml_ops):
    qml_device = qml.device("default.qubit", wires=nqubits)

    if qml_ops == "baseline":

        @qml.qnode(qml_device, interface="torch")
        def qnn_layer(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="Y")
            for layer_idx in range(nlayers):
                for wire in range(nqubits):
                    qml.RY(weights[layer_idx, wire], wires=wire)
                for wire in range(nqubits):
                    qml.CNOT(wires=(wire, (wire + 1) % nqubits))
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(nqubits))

        qnn_weight_shape = (nlayers, nqubits)
        return qnn_layer, qnn_weight_shape

    @qml.qnode(qml_device, interface="torch")
    def qnn_layer(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="X")
        for layer_idx in range(nlayers):
            for wire in range(nqubits):
                qml.RX(weights[layer_idx, wire, 0], wires=wire)
                qml.RZ(weights[layer_idx, wire, 1], wires=wire)
            for wire in range(nqubits):
                qml.CZ(wires=(wire, (wire + 1) % nqubits))
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(nqubits))

    qnn_weight_shape = (nlayers, nqubits, 2)
    return qnn_layer, qnn_weight_shape


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qnn_layer, qnn_weight_shape = build_qnn_layer(
        args.nqubits, args.nlayers, args.qml_ops
    )
    model = SimpleCNNQML(
        qnn_layer=qnn_layer,
        qnn_weight_shape=qnn_weight_shape,
        nqubits=args.nqubits,
    ).to(device)

    _, test_loader, train_dataset, test_dataset = mnist_loader(
        args.batch_size, args.num_workers, device.type == "cuda"
    )
    results_path = os.path.join("outputs", args.exp_name)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        n_folds=args.n_folds,
        epochs_per_fold=args.epochs_per_fold,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        train_dataset=train_dataset,
        criterion=criterion,
        device=device,
        results_path=results_path,
    )

    if args.fit:
        trainer.fit()
    trainer.evaluate(test_loader, test_dataset)


if __name__ == "__main__":
    main()
