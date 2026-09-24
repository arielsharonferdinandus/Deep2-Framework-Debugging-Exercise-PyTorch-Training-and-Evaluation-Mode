"""
Deep Learning Session 4 — Framework Debugging Exercise.

Experiment: Pola standar training, sebelum dan sesudah model.eval()

hal yang di demonstrasikan disini:
1. `model.eval()` hanya mengubah flag .training.
2. backward & optimizer tetap jalan.
3. `model.eval()` mengubah perilaku layer, namun gradient tetap terhitung.
4. `model.eval()` dengan `torch.no_grad()`.
"""

import torch
from torch import nn

def demo_training_before_after_eval():
    print("flag .training sebelum eval =>", model.training, [m.training for m in model])
    model.eval()
    print("flag .training setelah eval =>", model.training, [m.training for m in model])
    loss = nn.functional.mse_loss(model(X), y)
    loss.backward()
    print("requires_grad semua parameter:", all(p.requires_grad for p in model.parameters()))
    print("gradient tetap terhitung     :", all(p.grad is not None for p in model.parameters()))
    print("nilai grad layer akhir :", model[-1].weight.grad.flatten()[:3])

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        out_b = model(X)
        print("requires_grad output          :", out_b.requires_grad)
        print("grad_fn output                :", out_b.grad_fn)
        try:
            loss_b = nn.functional.mse_loss(out_b, y)
            loss_b.backward()
        except RuntimeError as e:
            print("backward() gagal, error       :", e)
    print("gradient tercatat?             :", any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 1))
    X = torch.randn(4, 2)
    y = torch.randn(4, 1)
    demo_training_before_after_eval()
