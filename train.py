import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.dynamic_resnet import DynamicResNet18
from utils import log_gate_usage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

initial_tau = 1.0
final_tau = 0.1
anneal_rate = 0.95

def get_tau(epoch):
    return max(final_tau, initial_tau * (anneal_rate ** epoch))

def set_tau(model, new_tau):
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            if hasattr(block, 'tau'):
                block.tau = new_tau

model = DynamicResNet18(use_controller=True, tau=initial_tau).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    print(f"\nStarting Epoch {epoch+1}")
    model.train()
    total, correct = 0, 0

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(1)
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

    log_gate_usage(model)

