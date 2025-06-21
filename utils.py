import torch
from models.dynamic_resnet import GatedBlock 

def log_gate_usage(model):
    usage = {}
    for name, module in model.named_modules():
        if isinstance(module, GatedBlock) and hasattr(module, "gate_history"):
            if module.gate_history:
                usage[name] = sum(module.gate_history) / len(module.gate_history)
                module.gate_history = []  # reset for next epoch
    print("Gate usage per block:")
    for name, val in usage.items():
        print(f"  {name}: {val:.4f}")

# Deprecated, as this was used for static gates 
def compute_gate_activation(model):
    gate_values = []
    for module in model.modules():
        if hasattr(module, "gate"):
            gate_values.append(torch.sigmoid(module.gate).item())
    avg_gate = sum(gate_values) / len(gate_values)
    print(f"Avg Gate Activation: {avg_gate:.4f} ({(1 - avg_gate)*100:.1f}% sparsity)")
    return avg_gate, gate_values
