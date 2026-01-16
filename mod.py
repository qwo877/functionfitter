import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_size=64, activation='tanh'):
        super(MLP, self).__init__()
        self.activation = self.get_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            self.activation,              
            nn.Linear(hidden_size, hidden_size),
            self.activation,              
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)
    
    def get_activation(self, activation):
        if isinstance(activation, str):
            activation = activation.lower()
            if activation == 'tanh':
                return nn.Tanh()
            elif activation == 'relu':
                return nn.ReLU()
            elif activation == 'sigmoid':
                return nn.Sigmoid()
            elif activation == 'softplus':
                return nn.Softplus()
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
        elif isinstance(activation, nn.Module):
            return activation
        else:
            raise TypeError("Activation must be a string or an instance of nn.Module")