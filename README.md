# Dynamic ResNet with Gated Sparsity and Knowledge Distillation

This project explores dynamic sparsity in convolutional neural networks using a gated ResNet-18 model trained on CIFAR-10. The goal is to reduce compute (FLOPs) during inference without a significant drop in accuracy.

## Approach

- **Dynamic Gating:** Each residual block is augmented with a learnable gate, controlled by a Gumbel-softmax mechanism, allowing the network to selectively skip blocks based on input.
- **Sparsity Penalty:** A regularization term encourages the network to minimize active gates to reduce computational cost.
- **Knowledge Distillation:** A pre-trained ResNet-18 teacher guides the training of the sparse student model to retain accuracy.
- **Exponential Annealing:** Gumbel temperature and distillation weighting are annealed across epochs to stabilize training.

## Results

- **Baseline ResNet-18**: 85% accuracy with ~37.5 MFLOPs.
- **Sparse + KD Model**: 70% accuracy with ~12–16 MFLOPs after 10 epochs.
- Achieves ~60% reduction in FLOPs with ~15% accuracy trade-off.

## Limitations

- Accuracy gap remains despite distillation and annealing.
- Gumbel-softmax introduces noise during training, leading to instability.
- Joint optimization of gates and weights is challenging and slow to converge.

## Future Work

- Explore hard routing or straight-through estimators.
- Decouple gate learning from weight training.
- Fine-tune sparse models after gate convergence.
- Apply to larger datasets (e.g., ImageNet) and other architectures.
