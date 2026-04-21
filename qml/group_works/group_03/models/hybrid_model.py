import torch
import torch.nn as nn
from pennylane.qnn import TorchLayer

class HybridModel(nn.Module):
    def __init__(self, num_qubits, quantum_circuit, weight_shapes):
        super(HybridModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Feature Reducer to match 16 amplitudes for 4 qubits
        self.feature_reducer = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 16), 
            nn.Softmax(dim=1)
        )
        
        self.qlayer = TorchLayer(quantum_circuit, weight_shapes)
        
        self.head = nn.Sequential(
            nn.Linear(num_qubits, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.feature_reducer(x)
        x = self.qlayer(x)
        return self.head(x)

