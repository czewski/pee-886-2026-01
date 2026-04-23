import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pennylane as qml

from qml.group_works.group_03.trainer import Trainer
from qml.group_works.group_03.models.hybrid_model import HybridModel

def mnist_loader(batch_size):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs-per-fold", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--exp-name", type=str, default="run_hybrid")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_qubits = args.num_qubits
    num_layers = args.num_layers
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
        qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    weight_shapes = {"weights": (num_layers, num_qubits)}
    model = HybridModel(num_qubits, quantum_circuit, weight_shapes).to(device)

    train_loader, test_loader, train_dataset, test_dataset = mnist_loader(args.batch_size)

    results_path = os.path.join("outputs", args.exp_name)
    os.makedirs(results_path, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        n_folds=args.n_folds,
        epochs_per_fold=args.epochs_per_fold,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        seed=42,
        num_workers=0,
        train_dataset=train_dataset,
        criterion=criterion,
        device=device,
        results_path=results_path,
    )

    trainer.fit()
    trainer.evaluate(test_loader, test_dataset)


if __name__ == "__main__":
    main()
