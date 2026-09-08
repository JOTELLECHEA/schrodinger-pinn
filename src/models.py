import torch
import torch.nn as nn

class QuantumPINN(nn.Module):
    def __init__(self, hidden_dim: int = 50, depth: int = 4, e_init: float = 1.0):
        super().__init__()

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
        self.E = nn.Parameter(torch.tensor([e_init]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QuantumPINN2D(nn.Module):
    def __init__(self, hidden_dim: int = 50, depth: int = 4, e_init: float = 1.0):
        super().__init__()

        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
        self.E = nn.Parameter(torch.tensor([e_init]))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)