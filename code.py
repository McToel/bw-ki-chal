import torch
import numpy as np
import torch.nn as nn


def train(D):

    torch.manual_seed(0)
    # Hyper-parameter
    n_steps = 11343 #8350#5900#1465
    learning_rate = 0.0001 #0.0001005#0.000211
    input_size = 9
    output_size = 1

    # Trainings-Daten vorbereiten
    X = D[:, :-1].astype(np.float32)
       
    # The 14th column is the result column:
    y = D[:, -1].astype(np.float32)
    X_train = torch.from_numpy(X)
    y_train = torch.from_numpy(y)
    feature_means = (X_train[y_train[:] == 1, :].mean(dim=0) + X_train[y_train[:] == 0, :].mean(dim=0)) / 2

    # Modell definieren
    class MLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            neuron = 7
            self.layers = nn.Sequential(
                nn.Linear(input_size, neuron),
                nn.PReLU(),
                nn.Linear(neuron, output_size)
            )

        def forward(self, x):
            x = x - feature_means
            x = x[:, [1,2,4,7,8,9,10,11,12]]
            out = self.layers(x)
            return out

    model = MLP(input_size)

    # loss and optimizer
    # checkout: https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()  # sigmoid + binary cross entropy

    # optimizer
    # Dokumentation: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Alternative zu Adam
    # Dokumentation: https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
    # momentum = 0.9  # Wert zwischen 0. und 1.
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    cost = np.zeros(n_steps)
    acc_t = np.zeros(n_steps)
    # trainieren des Modells
    for e in range(n_steps):
        # forward pass
        outputs = model.forward(X_train)[:, 0]  # Xw (linear layer)
        loss = criterion(outputs, y_train)  # sigmoid and cross-entropy loss
        cost[e] = loss

        # backward pass (automatically computes gradients)
        optimizer.zero_grad()  # reset gradients (torch accumulates them)
        loss.backward()  # computes gradients

        # Optimierungsschritt durchfuehren
        optimizer.step()

return model
