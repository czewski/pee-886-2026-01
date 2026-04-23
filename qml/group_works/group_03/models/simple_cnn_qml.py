import torch
import torch.nn as nn
from pennylane.qnn import TorchLayer


class SimpleCNNQML(nn.Module):
    def __init__(self, qnn_layer, qnn_weight_shape, nqubits=4):
        super().__init__()
        self.nqubits = nqubits
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.feature_reducer = nn.Linear(16 * 7 * 7, nqubits)
        self.qlayer = TorchLayer(qnn_layer, {"weights": qnn_weight_shape})
        self.output_layer = nn.Linear(nqubits, 10)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.feature_reducer(x)
        x = torch.tanh(x) * torch.pi
        x = self.qlayer(x)
        if x.ndim == 3 and x.shape[0] == self.nqubits:
            x = x.permute(1, 0, 2)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return self.output_layer(x)
