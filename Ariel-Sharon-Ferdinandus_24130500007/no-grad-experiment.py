"""
Deep Learning Session 4 — Framework Debugging Exercise.

Experiment: Perbedaan perilaku .train dan .eval pada mode dropout dan batchnorm

hal yang di demonstrasikan disini:
1. Seberapa penting/dibutuhkannya `no_grad()`
2. Perbedaan Memori yang digunakan menggunakan `no_grad()`
3. Kecepeatan komputasi saat menggunakan `no_grad()`
"""

import torch
from torch import nn
import tracemalloc
import time

def demo_no_grad_benefits():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(100, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 10),
    )
    X = torch.randn(256, 100)

    out_with_grad = model(X)
    print("requires_grad:", out_with_grad.requires_grad,
          "| grad_fn:", out_with_grad.grad_fn is not None)

    with torch.no_grad():
        out_no_grad = model(X)
    print("requires_grad:", out_no_grad.requires_grad,
          "| grad_fn:", out_no_grad.grad_fn)

    tracemalloc.start()
    _ = model(X)  # TANPA no_grad ->  backward
    _, peak_with_grad = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    with torch.no_grad():
        _ = model(X)  # DENGAN no_grad -> cannot backward
    _, peak_no_grad = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"peak memory          : {peak_with_grad / 1024:.1f} KB")
    print(f"peak memory no_grad(): {peak_no_grad / 1024:.1f} KB")

    n_runs = 50

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(X)
    waktu_dengan_grad = time.perf_counter() - start

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(X)
    waktu_no_grad = time.perf_counter() - start

    print(f"waktu          : {waktu_dengan_grad*1000:.2f} ms total")
    print(f"waktu no_grad(): {waktu_no_grad*1000:.2f} ms total")

    with torch.no_grad():
        out = model(X)
        try:
            out.sum().backward()
        except RuntimeError as e:
            print("RuntimeError expected:")
            print(" ", e)

if __name__ == "__main__":
    demo_no_grad_benefits()
