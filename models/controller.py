import torch
import torch.nn as nn

class GateController(nn.Module):
    def __init__(self, in_channels: int, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
