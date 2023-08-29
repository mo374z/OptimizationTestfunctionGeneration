import torch as torch

class NN(torch.nn.Module):
    def __init__(self, n_in, n_hidden, hidden_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(torch.nn.Linear(n_in, n_hidden))
            self.layers.append(torch.nn.ReLU())
            n_in = n_hidden

        # delete last ReLU
        #self.layers = self.layers[:-1]
        self.layers.append(torch.nn.Linear(n_hidden, 1))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class NN_(torch.nn.Module):
    def __init__(self, n_in, n_out, n_hidden, hidden_layers, activation=torch.nn.ReLU()):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(torch.nn.Linear(n_in, n_hidden))
            self.layers.append(activation)
            n_in = n_hidden

        # delete last ReLU
        #self.layers = self.layers[:-1]
        self.layers.append(torch.nn.Linear(n_hidden, n_out))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NN_norm(torch.nn.Module):
    def __init__(self, n_in, n_hidden, hidden_layers, activation=torch.nn.ReLU):
        super().__init__()

        # Define the normalization layer
        self.normalize = torch.nn.BatchNorm1d(n_in)

        self.layers = torch.nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(torch.nn.Linear(n_in, n_hidden))
            self.layers.append(activation)
            n_in = n_hidden

        self.layers.append(torch.nn.Linear(n_hidden, 1))

    def forward(self, x):
        # Apply input normalization
        x = self.normalize(x)

        for layer in self.layers:
            x = layer(x)
        return x


# create class for NN with outputdim = 1
class NN1(torch.nn.Module):
    def __init__(self, n_in, n_hidden, hidden_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(torch.nn.Linear(n_in, n_hidden))
            self.layers.append(torch.nn.ReLU())
            n_in = n_hidden

        # delete last ReLU
        self.layers = self.layers[:-1]
        self.layers.append(torch.nn.Linear(n_hidden, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class derivative_NN(torch.nn.Module):
    def __init__(self, n_in, n_hidden, hidden_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(torch.nn.Linear(n_in, n_hidden))
            self.layers.append(torch.nn.ReLU())
            n_in = n_hidden

        # delete last ReLU
        self.layers = self.layers[:-1]
        self.layers.append(torch.nn.Linear(n_hidden, 1))