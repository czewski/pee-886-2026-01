import torch
import torch.nn as nn
from pennylane.qnn import TorchLayer
import torch.nn.functional as F


# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.flatten(start_dim=1)
#         x = self.class_head(x)
#         return x


class HybridClassifier(nn.Module):
    def __init__(self, qnn_layer, nqubits=4, nlayers=2):
        super().__init__()
        self.nqubits = nqubits
        # to convert image arrays to vectors
        self.flatten = nn.Flatten()
        # to reduce number of features for input to the qnn
        self.reduction_layer = nn.Linear(28 * 28, nqubits)
        self.qlayer = TorchLayer(qnn_layer, {"weights": (nlayers, nqubits)})
        # to map the qnn outputs to a class (some values are unused if
        # not using all 10 classes)
        self.output_layer = nn.Linear(nqubits, 10)

    def forward(self, x):
        # transform image array to a vector
        x = self.flatten(x)

        # apply classical dimensionality reduction layer
        x = self.reduction_layer(x)
        # apply pi*tanh activation to put data into the range from -pi
        # to pi
        x = torch.tanh(x) * torch.pi
        x = self.qlayer(x)
        if x.ndim == 3 and x.shape[0] == self.nqubits:
            x = x.permute(1, 0, 2)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        # apply output layer to combine qnn outputs to 10 numbers
        x = self.output_layer(x)
        # just return outputs since we will use cross entropy loss in
        # training
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        # Same convolutions -> output -> same as input dimension

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        # last conv out was 8 so this conv input is 8.

        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
        # in fc1 -> 16 bcoz that is outchannel in conv2 and 7*7 because -> 2 poolings will make it (28/2)/2 => 7

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
