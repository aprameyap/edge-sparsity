import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from models.dynamic_resnet import DynamicResNet18
from utils import log_gate_usage, compute_flops, get_sparsity_lambda
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--use_kd', action='store_true', help='Enable knowledge distillation')
parser.add_argument('--epochs', type=int, default=3)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
usage_log = {}

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

initial_tau = 1.0
final_tau = 0.1
decay_rate = 0.8

def get_tau(epoch):
    return final_tau + (initial_tau - final_tau) * math.exp(-decay_rate * epoch)

def set_tau(model, new_tau):
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            if hasattr(block, 'tau'):
                block.tau = new_tau

def kd_loss(student_logits, teacher_logits, true_labels, alpha=0.7, T=4.0):
    ce_loss = F.cross_entropy(student_logits, true_labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    return alpha * soft_loss + (1 - alpha) * ce_loss

model = DynamicResNet18(use_controller=True, tau=initial_tau).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

if args.use_kd:
    teacher = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    teacher.fc = nn.Linear(512, 10)
    teacher = teacher.to(device)
    teacher.eval()
else:
    criterion = nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    print(f"\nStarting Epoch {epoch+1}")
    model.train()
    total, correct = 0, 0

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        student_outputs = model(images)

        if args.use_kd:
            with torch.no_grad():
                teacher_outputs = teacher(images)
            loss = kd_loss(student_outputs, teacher_outputs, labels)
        else:
            loss = criterion(student_outputs, labels)

        # Sparsity penalty
        sparsity_penalty = 0.0
        gate_count = 0
        for module in model.modules():
            if hasattr(module, "gate_value_log") and module.gate_value_log is not None:
                sparsity_penalty += module.gate_value_log
                gate_count += 1
        if gate_count > 0:
            sparsity_penalty /= gate_count
            loss += get_sparsity_lambda(epoch) * sparsity_penalty

        loss.backward()
        optimizer.step()

        pred = student_outputs.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        if batch_idx % 100 == 0:
            acc = correct / total * 100
            print(f"[Batch {batch_idx}] Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

    acc = correct / total
    print(f"Epoch {epoch+1} complete. Train Accuracy: {acc:.4f}")

    new_tau = get_tau(epoch + 1)
    set_tau(model, new_tau)
    print(f"Updated Gumbel temperature (tau): {new_tau:.4f}")

    log_gate_usage(model, usage_log, epoch + 1)

    model.eval()
    total_flops, _ = compute_flops(model)
    print(f"[FLOPs] Dynamic Model: {total_flops / 1e6:.2f} MFLOPs")

all_layers = sorted(set(k for v in usage_log.values() for k in v))
for layer in all_layers:
    x, y = [], []
    for epoch in sorted(usage_log.keys()):
        x.append(epoch)
        y.append(usage_log[epoch].get(layer, 0.0))
    plt.plot(x, y, label=layer)

plt.xlabel("Epoch")
plt.ylabel("Gate Usage (Average)")
plt.title("Gate Usage per Layer Over Epochs")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.savefig("gate_usage_plot.png", dpi=200)
plt.show()
