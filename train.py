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
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--multi-gpu', action='store_true', help='Enable multi-GPU training using DataParallel')
parser.add_argument('--gpu-ids', type=str, default='0,1,2,3', help='GPU IDs to use (comma-separated), e.g., "0,1,2,3"')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU (default: 64)')
args = parser.parse_args()

def setup_device_and_gpus():
    """Setup device and GPU configuration"""
    if args.multi_gpu and torch.cuda.is_available():
        # Parse GPU IDs
        gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
        
        # Check if all specified GPUs are available
        available_gpus = torch.cuda.device_count()
        gpu_ids = [id for id in gpu_ids if id < available_gpus]
        
        if len(gpu_ids) == 0:
            print("No valid GPUs found. Falling back to single GPU/CPU.")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            use_multi_gpu = False
            gpu_ids = []
        else:
            device = torch.device(f"cuda:{gpu_ids[0]}")
            use_multi_gpu = True
            print(f"Using GPUs: {gpu_ids}")
            print(f"Primary device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_multi_gpu = False
        gpu_ids = []
        print(f"Using single device: {device}")
    
    return device, use_multi_gpu, gpu_ids

device, use_multi_gpu, gpu_ids = setup_device_and_gpus()
usage_log = {}

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

effective_batch_size = args.batch_size
if use_multi_gpu:
    total_batch_size = args.batch_size * len(gpu_ids)
    print(f"Effective batch size: {args.batch_size} per GPU × {len(gpu_ids)} GPUs = {total_batch_size}")
else:
    print(f"Batch size: {effective_batch_size}")

trainloader = DataLoader(trainset, batch_size=effective_batch_size, shuffle=True, 
                        num_workers=4, pin_memory=True)

# Temperature scheduling
initial_tau = 1.0
final_tau = 0.1
decay_rate = 0.8

def get_tau(epoch):
    return final_tau + (initial_tau - final_tau) * math.exp(-decay_rate * epoch)

def set_tau(model, new_tau):
    # Handle both single model and DataParallel wrapped model
    model_to_update = model.module if hasattr(model, 'module') else model
    
    for layer in [model_to_update.layer1, model_to_update.layer2, model_to_update.layer3, model_to_update.layer4]:
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

def calculate_sparsity_penalty(model):
    """Calculate sparsity penalty from gate values"""
    sparsity_penalty = 0.0
    gate_count = 0
    
    model_to_check = model.module if hasattr(model, 'module') else model
    
    for module in model_to_check.modules():
        if hasattr(module, "gate_value_log") and module.gate_value_log is not None:
            sparsity_penalty += module.gate_value_log
            gate_count += 1
    
    if gate_count > 0:
        sparsity_penalty /= gate_count
    
    return sparsity_penalty

model = DynamicResNet18(use_controller=True, tau=initial_tau).to(device)

if use_multi_gpu:
    print(f"Wrapping model with DataParallel for GPUs: {gpu_ids}")
    model = nn.DataParallel(model, device_ids=gpu_ids)

optimizer = optim.Adam(model.parameters(), lr=0.001)

if args.use_kd:
    teacher = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    teacher.fc = nn.Linear(512, 10)
    teacher = teacher.to(device)
    
    if use_multi_gpu:
        teacher = nn.DataParallel(teacher, device_ids=gpu_ids)
    
    teacher.eval()
    print("Knowledge distillation enabled with ResNet18 teacher")
else:
    criterion = nn.CrossEntropyLoss()
    print("Training without knowledge distillation")

print(f"Starting training for {args.epochs} epochs...")

for epoch in range(args.epochs):
    print(f"\n{'='*50}")
    print(f"Starting Epoch {epoch+1}/{args.epochs}")
    print(f"{'='*50}")
    
    model.train()
    total, correct = 0, 0
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        student_outputs = model(images)
        
        if args.use_kd:
            with torch.no_grad():
                teacher_outputs = teacher(images)
            loss = kd_loss(student_outputs, teacher_outputs, labels)
        else:
            loss = criterion(student_outputs, labels)
        
        sparsity_penalty = calculate_sparsity_penalty(model)
        if sparsity_penalty > 0:
            loss += get_sparsity_lambda(epoch) * sparsity_penalty
        
        loss.backward()
        optimizer.step()
        
        pred = student_outputs.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            acc = correct / total * 100
            avg_loss = running_loss / (batch_idx + 1)
            print(f"[Batch {batch_idx:4d}/{len(trainloader)}] "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | "
                  f"Sparsity: {sparsity_penalty:.4f}")
    
    acc = correct / total
    epoch_loss = running_loss / len(trainloader)
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Average Loss: {epoch_loss:.4f}")
    
    new_tau = get_tau(epoch + 1)
    set_tau(model, new_tau)
    print(f"Updated Gumbel temperature (tau): {new_tau:.4f}")
    
    log_gate_usage(model, usage_log, epoch + 1)
    
    model.eval()
    try:
        model_for_flops = model.module if hasattr(model, 'module') else model
        total_flops, *_ = compute_flops(model_for_flops)
        print(f"[FLOPs] Dynamic Model: {total_flops / 1e6:.2f} MFLOPs")
    except Exception as e:
        print(f"[Warning] Could not compute FLOPs: {e}")

print(f"\n{'='*50}")
print("Training completed!")
print(f"{'='*50}")

print("Generating gate usage plot...")
all_layers = sorted(set(k for v in usage_log.values() for k in v))

if all_layers:
    plt.figure(figsize=(12, 8))
    for layer in all_layers:
        x, y = [], []
        for epoch in sorted(usage_log.keys()):
            x.append(epoch)
            y.append(usage_log[epoch].get(layer, 0.0))
        plt.plot(x, y, label=layer, marker='o')
    
    plt.xlabel("Epoch")
    plt.ylabel("Gate Usage (Average)")
    plt.title("Gate Usage per Layer Over Epochs")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig("gate_usage_plot.png", dpi=200, bbox_inches='tight')
    plt.show()
    print("Gate usage plot saved as 'gate_usage_plot.png'")
else:
    print("No gate usage data to plot.")

# Save model
# model_to_save = model.module if hasattr(model, 'module') else model
# save_name = f"dynamic_resnet_{'kd_' if args.use_kd else ''}{'multi_gpu' if use_multi_gpu else 'single_gpu'}.pth"
# torch.save(model_to_save.state_dict(), save_name)
# print(f"Model saved as '{save_name}'")