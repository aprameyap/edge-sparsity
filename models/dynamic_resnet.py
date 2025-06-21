import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.controller import GateController, gumbel_softmax

class GatedBlock(nn.Module):
    def __init__(self, block: nn.Module, use_controller=False, tau=1.0):
        super().__init__()
        self.block = block
        self.downsample = block.downsample if hasattr(block, "downsample") and block.downsample else None
        self.use_controller = use_controller
        self.tau = tau
        self.gate_history = []

        if use_controller:
            self.controller = nn.Sequential(
                GateController(block.conv1.in_channels),
                nn.Linear(1, 2)  # Binary classification: skip or activate
            )
        else:
            self.gate = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        if self.use_controller:
            logits = self.controller[0](x)  # (B, 1)
            logits = self.controller[1](logits)  # (B, 2)
            gate = gumbel_softmax(logits, tau=self.tau, hard=True)[:, 1:2]  # Use 2nd class for "activate"
            gate = gate.view(-1, 1, 1, 1)
            self.gate_history.append(gate.mean().item())
        else:
            gate = torch.sigmoid(self.gate)

        return gate * self.block(x) + (1 - gate) * residual

class DynamicResNet18(nn.Module):
    def __init__(self, num_classes=10, use_controller=False, tau=1.0):
        super().__init__()
        base_model = nn.Sequential(*list(nn.Sequential(*list(models.resnet18(weights=None).children()))[:-2]))
        model_layers = list(models.resnet18(weights=None).children())

        self.initial = nn.Sequential(model_layers[0], model_layers[1], model_layers[2], model_layers[3])
        self.layer1 = self._wrap_layer(model_layers[4], use_controller, tau)
        self.layer2 = self._wrap_layer(model_layers[5], use_controller, tau)
        self.layer3 = self._wrap_layer(model_layers[6], use_controller, tau)
        self.layer4 = self._wrap_layer(model_layers[7], use_controller, tau)
        self.avgpool = model_layers[8]
        self.fc = nn.Linear(model_layers[9].in_features, num_classes)

    def _wrap_layer(self, layer, use_controller, tau):
        return nn.Sequential(*[GatedBlock(b, use_controller, tau) for b in layer])

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)