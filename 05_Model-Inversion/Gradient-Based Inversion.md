# Gradient-Based Inversion

**Tags:** #model-inversion #gradient-leakage #privacy #reconstruction #ai-red-teaming

---

## Overview

**Gradient-based model inversion** leverages exposed gradients (especially during training or fine-tuning) to reconstruct input data — often images, text, or other private content. This is a privacy risk in **federated learning**, **multi-tenant models**, and during collaborative fine-tuning.

Key idea:

> Gradients carry partial information about the training input — with enough optimization, an attacker can invert the gradients and recover the input that caused them.

---

## Attack Pre-Conditions

* **White-box** or **semi-white-box** access
* Access to:

  * Model architecture
  * Weights (optional)
  * **Gradients from training updates**
* Optional:

  * Partial label or batch index info
  * Training set statistics (mean, std)

---

## Core Method: Optimization-Based Gradient Inversion

Reconstruct an input `x'` such that:

```math
Grad(model(x')) ≈ Observed_Gradients
```

### Step-by-step:

1. **Initialize dummy input (`x'`)** — random noise.
2. **Forward pass** through model.
3. **Backprop** and compare generated gradients with target gradients.
4. **Iteratively update `x'`** using gradient descent to minimize loss.

---

## Python Example (Image Inversion)

```python
import torch
from torch.nn import functional as F

# Assume observed_grad is from a target model training step
observed_grad = ...

# Dummy input to optimize
x_prime = torch.randn((1, 3, 224, 224), requires_grad=True)

optimizer = torch.optim.Adam([x_prime], lr=0.1)

for _ in range(1000):
    optimizer.zero_grad()
    output = model(x_prime)
    loss = F.cross_entropy(output, label)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    
    # Compare with observed gradients
    loss_grad = sum(((g1 - g2)**2).sum() for g1, g2 in zip(grads, observed_grad))
    loss_grad.backward()
    optimizer.step()
```

---

## Real-World Applications

| Scenario                        | Risk                                               |
| ------------------------------- | -------------------------------------------------- |
| Federated learning (FL)         | Clients can leak gradients, revealing private data |
| Collaborative model fine-tuning | Shared gradient updates can leak raw inputs        |
| Multi-tenant training servers   | A malicious user can record gradients              |
| Debugging/training pipelines    | Accidental gradient leaks                          |

---

## Popular Implementations

* **Deep Leakage from Gradients (DLG)**
  [https://github.com/mit-han-lab/dlg](https://github.com/mit-han-lab/dlg)

* **iDLG** (Improved DLG with label inference)
  [https://arxiv.org/abs/2001.02610](https://arxiv.org/abs/2001.02610)

---

## Key Tools & Libraries

| Tool                  | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `dlg`                 | PyTorch-based Deep Leakage from Gradients repo       |
| `privacy-leakage`     | FL gradient attack demos                             |
| `Opacus`              | Differential Privacy engine for PyTorch              |
| `TensorFlow Privacy`  | Gradient clipping, noise addition                    |
| `FedTorch` / `Flower` | Federated learning frameworks for simulation/testing |

---

## Mitigation Techniques

| Defense                   | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| **Differential privacy**  | Add noise to gradients (e.g., Opacus, TF Privacy)    |
| **Gradient clipping**     | Limits exposure from large updates                   |
| **Update filtering**      | Server ignores suspiciously informative updates      |
| **Encrypted computation** | Homomorphic encryption or secure aggregation         |
| **Split learning**        | Data remains on device; model is split               |
| **Label hiding**          | Prevents attacker from matching gradients to classes |

---

## Visualization Example

| Gradient           | Reconstructed Input                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------ |
| From training step | ![x](https://raw.githubusercontent.com/mit-han-lab/dlg/master/resources/imagenet_demo.png) |

*Source: MIT Han Lab — DLG repository*

---

## References

* **Zhu et al.** "Deep Leakage from Gradients" – NeurIPS 2019
  [https://arxiv.org/abs/1906.08935](https://arxiv.org/abs/1906.08935)

* **Zhao et al.** "iDLG: Improved Deep Leakage"
  [https://arxiv.org/abs/2001.02610](https://arxiv.org/abs/2001.02610)

* **Opacus (Meta)**
  [https://github.com/pytorch/opacus](https://github.com/pytorch/opacus)

---

## Commands and Setup (Opacus)

```bash
pip install opacus
```

```python
from opacus import PrivacyEngine

model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)
```

---

