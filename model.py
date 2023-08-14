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


class NN_norm(torch.nn.Module):
    def __init__(self, n_in, n_hidden, hidden_layers):
        super().__init__()

        # Define the normalization layer
        self.normalize = torch.nn.BatchNorm1d(n_in)

        self.layers = torch.nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(torch.nn.Linear(n_in, n_hidden))
            self.layers.append(torch.nn.ReLU())
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

# class for surrogate model
class SurrogateModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SurrogateModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 =  torch.nn.Linear(hidden_dim, hidden_dim)
        self.tanh3 = torch.nn.Tanh()
        self.fc4 = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid4 = torch.nn.Sigmoid()
        self.fc5 = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.tanh3(self.fc3(x))
        x = self.sigmoid4(self.fc4(x))
        x = self.fc5(x)
        return x


# class for attention model
class AttentionNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionNN, self).__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=0)
        attended_input = torch.sum(x * attention_weights, dim=0)
        output = self.fc(attended_input)
        return output