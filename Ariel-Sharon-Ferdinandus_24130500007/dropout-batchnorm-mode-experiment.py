"""
Deep Learning Session 4 — Framework Debugging Exercise.

Experiment: Perbedaan perilaku .train dan .eval pada mode dropout dan batchnorm

hal yang di demonstrasikan disini:
1. Berbeda dengan `.train()` dropout tidak aktif pada  `.eval()`
2. Begitu pula pada batchnorm tidak menormalisasikan pada `.eval()` melainkan Identitas
"""

import torch
from torch import nn

def demo_train_eval_behavior_on_dropout():
    drop = nn.Dropout(p=0.5)
    x = torch.ones(1, 10)

    drop.train()
    print("1st train:", drop(x))
    print("2nd train:", drop(x))

    drop.eval()
    print("1st eval:", drop(x))
    print("2nd eval:", drop(x))

def demo_train_eval_behavior_on_batchnorm():
    torch.manual_seed(0)
    X = torch.randn(256, 4) * 3 + 10  # data dengan mean ~10, std ~3 (var ~9)
 
    for mode in ("eval", "train"):
        bn = nn.BatchNorm1d(4)
        for _ in range(20):
            getattr(bn, mode)()
            bn(X)
        print(f"\n[training dilakukan dalam mode {mode}]")
        print("running_mean       :", bn.running_mean)
        print("running_var        :", bn.running_var)
        print("num_batches_tracked:", bn.num_batches_tracked.item())

        bn.eval()
        with torch.no_grad():
            out = bn(X)
        print(f"output saat eval: mean={out.mean():.3f}, std={out.std():.3f}")

if __name__ == "__main__":
    demo_train_eval_behavior_on_dropout()
    demo_train_eval_behavior_on_batchnorm()
