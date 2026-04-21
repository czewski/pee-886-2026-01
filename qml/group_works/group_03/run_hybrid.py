import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pennylane as qml
from pennylane.qnn import TorchLayer
import time

import qml.group_works.group_03.trainer as trainer
from qml.group_works.group_03.hybrid_model import HybridModel

def mnist_loader():
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_qubits = 4
num_layers = 2
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
    qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

weight_shapes = {"weights": (num_layers, num_qubits)}

model = HybridModel(num_qubits, quantum_circuit, weight_shapes).to(device)

train_loader, test_loader, train_dataset, test_dataset = mnist_loader()

n_folds = 5
epochs_per_fold = 50
patience = 15

results_path = os.path.join(os.getcwd(), 'results')
os.makedirs(results_path, exist_ok=True)

criterion = nn.CrossEntropyLoss()

trainer = trainer.Trainer(model, n_folds, epochs_per_fold, patience, train_dataset, criterion, device, results_path)

trainer.fit()
trainer.evaluate(test_loader, test_dataset)