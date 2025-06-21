import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.dynamic_resnet import DynamicResNet18
from utils import compute_gate_activation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

model = DynamicResNet18(use_controller=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
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

        sparsity_penalty = 0
        for module in model.modules():
            if hasattr(module, "gate"):
                sparsity_penalty += torch.abs(torch.sigmoid(module.gate)).mean()
        
        loss += loss + (0.01 * sparsity_penalty)

        # Log every 100 batches
        if batch_idx % 100 == 0:
            acc = correct / total * 100
            print(f"[Batch {batch_idx}] Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

    print(f"Epoch {epoch+1} complete. Train Accuracy: {correct/total:.4f}")