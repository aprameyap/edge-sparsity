import torch
import torch.nn as nn
import torch.nn.functional as F

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

def gumbel_softmax(logits, tau=1.0, hard=False):
    gumbel_noise = -torch.empty_like(logits).exponential_().log()
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1) if not hard else F.one_hot(y.argmax(dim=-1), num_classes=logits.shape[-1]).float()    
