import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pennylane as qml
import numpy as np

from qml.group_works.group_03.trainer import Trainer
from qml.group_works.group_03.models.linear import HybridClassifier, CNN
from pennylane.workflow import get_best_diff_method

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
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


def main():
    args = parse_args()
    set_seed(args.seed)

    nqubits = args.nqubits
    nlayers = args.nlayers
    qml_device = qml.device("lightning.qubit", wires=nqubits)

    # Here we define the quantum part of the model
    @qml.qnode(qml_device, interface="torch")
    def qnn_layer(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(nqubits), rotation="X")
        # place gates using the trainable weights
        # as parameters.
        for layer_index in range(nlayers):
            # place the trainable rotations
            for i in range(nqubits):
                qml.RY(weights[layer_index, i], wires=i)
            # place the entangling gates
            for i in range(nqubits):
                j = (i + 1) % nqubits
                qml.CNOT(wires=(i, j))
        # now, return the pauli Z expectation values
        # on each qubit.
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(nqubits))

    model = HybridClassifier(qnn_layer, nqubits, nlayers).to(device)
    # model = CNN(in_channels=1, num_classes=10).to(device)

    _, test_loader, train_dataset, test_dataset = mnist_loader(
        args.batch_size, args.num_workers, device.type == "cuda"
    )

    results_path = os.path.join("./results")
    os.makedirs(results_path, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model,
        args.n_folds,
        args.epochs_per_fold,
        args.patience,
        args.batch_size,
        args.learning_rate,
        args.seed,
        args.num_workers,
        train_dataset,
        criterion,
        device,
        results_path,
    )

    if args.fit:
        trainer.fit()
    trainer.evaluate(test_loader, test_dataset)


if __name__ == "__main__":
    main()
