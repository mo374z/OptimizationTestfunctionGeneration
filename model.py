import torch as torch

class NN(torch.nn.Module):
    def __init__(self, n_in, n_hidden, hidden_layers, activation=torch.nn.ReLU):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(torch.nn.Linear(n_in, n_hidden))
            self.layers.append(activation)
            n_in = n_hidden

        # delete last ReLU
        #self.layers = self.layers[:-1]
        self.layers.append(torch.nn.Linear(n_hidden, 1))


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
    
class NN3_norm(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.normalize = torch.nn.BatchNorm1d(input_dim)
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 =  torch.nn.Linear(hidden_dim, hidden_dim)
        self.tanh3 = torch.nn.Tanh()
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid4 = torch.nn.Sigmoid()
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.prelu = torch.nn.PReLU()
        self.fc6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.elu = torch.nn.ELU()
        self.fc7 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.prelu = torch.nn.PReLU()
        self.fc8 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.elu = torch.nn.ELU()
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    
    def forward(self, x):
        # Apply input normalization
        x = self.normalize(x)

        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.tanh3(self.fc3(x))
        x = self.sigmoid4(self.fc4(x))
        x = self.prelu(self.fc5(x))
        x = self.elu(self.fc6(x))
        x = self.prelu(self.fc7(x))
        x = self.elu(self.fc8(x))
        x = self.output_layer(x)
        return x

# Define a Graph Neural Network architecture
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(2, 16)  # Input dimension: 2, Output dimension: 16
        self.conv2 = GCNConv(16, 1)  # Output dimension: 1

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x.view(-1)


# class for surrogate model
class SurrogateModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5, weight_decay=0.001):
        super(SurrogateModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 =  torch.nn.Linear(hidden_dim, hidden_dim)
        self.tanh3 = torch.nn.Tanh()
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid4 = torch.nn.Sigmoid()
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.prelu = torch.nn.PReLU()
        self.fc6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.elu = torch.nn.ELU()
        self.fc7 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.prelu = torch.nn.PReLU()
        self.fc8 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.elu = torch.nn.ELU()
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.weight_decay = weight_decay

    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.tanh3(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid4(self.fc4(x))
        x = self.dropout(x)
        x = self.prelu(self.fc5(x))
        x = self.dropout(x)
        x = self.elu(self.fc6(x))
        x = self.dropout(x)
        x = self.prelu(self.fc7(x))
        x = self.dropout(x)
        x = self.elu(self.fc8(x))
        x = self.dropout(x)
        x = self.output_layer(x)
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
    

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(16, 1, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = torch.nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.residual_block1 = ResidualBlock(64, 64)
        self.residual_block2 = ResidualBlock(64, 64)
        self.fc = torch.nn.Linear(64 * 2, 1)  # Adjust input_length
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x