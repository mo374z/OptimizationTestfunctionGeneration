import torch as torch
import bo_torch as bo_torch

class NN(torch.nn.Module):
    def __init__(self, n_in, n_hidden, hidden_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(torch.nn.Linear(n_in, n_hidden))
            self.layers.append(torch.nn.ReLU())
            n_in = n_hidden

        # delete last ReLU
        self.layers = self.layers[:-1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Model(bo_torch.models.Model):
    def