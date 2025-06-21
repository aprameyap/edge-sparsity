import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from utils import compute_flops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

model = models.resnet18(weights=None, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
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

    print(f"Epoch {epoch+1} complete. Train Accuracy: {correct/total:.4f}")
    
    if epoch == 0:
        model.eval()
        total_flops, flops_by_module = compute_flops(model)
        print(f"[FLOPs] Total: {total_flops/1e6:.2f} MFLOPs")
