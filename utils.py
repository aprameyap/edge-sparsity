import torch

# Deprecated, as this was used for static gates 
def compute_gate_activation(model):
    gate_values = []
    for module in model.modules():
        if hasattr(module, "gate"):
            gate_values.append(torch.sigmoid(module.gate).item())
    avg_gate = sum(gate_values) / len(gate_values)
    print(f"Avg Gate Activation: {avg_gate:.4f} ({(1 - avg_gate)*100:.1f}% sparsity)")
    return avg_gate, gate_values
