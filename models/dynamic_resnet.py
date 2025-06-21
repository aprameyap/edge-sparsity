import torch
import torch.nn as nn
import torchvision.models as models

class GatedBlock(nn.Module):
    def __init__(self, block: nn.Module, use_controller=False):
        super().__init__()
        self.block = block
        self.use_controller = use_controller

        # Add downsampling if needed
        self.downsample = block.downsample if hasattr(block, "downsample") and block.downsample else None

        if use_controller:
            self.controller = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(block.conv1.in_channels, 1)
            )
        else:
            self.gate = nn.Parameter(torch.tensor([1.0]))  # Learnable scalar

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        if self.use_controller:
            gate_value = torch.sigmoid(self.controller(x))
        else:
            gate_value = torch.sigmoid(self.gate)
        return gate_value * self.block(x) + (1 - gate_value) * residual


class DynamicResNet18(nn.Module):
    def __init__(self, num_classes=10, use_controller=False):
        super().__init__()
        base_model = models.resnet18(weights=None)
        self.initial = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = self._wrap_layer(base_model.layer1, use_controller)
        self.layer2 = self._wrap_layer(base_model.layer2, use_controller)
        self.layer3 = self._wrap_layer(base_model.layer3, use_controller)
        self.layer4 = self._wrap_layer(base_model.layer4, use_controller)
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def _wrap_layer(self, layer, use_controller):
        return nn.Sequential(*[GatedBlock(b, use_controller) for b in layer])

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
