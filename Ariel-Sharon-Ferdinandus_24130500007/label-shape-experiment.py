"""
Deep Learning Session 4 — Framework Debugging Exercise.

Experiment: Perbedaan perilaku .train dan .eval pada mode dropout dan batchnorm

hal yang di demonstrasikan disini:
1. Mismatch Shape akan menghasilkan Error
"""

import torch
from torch import nn

def demo_label_shape():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.randn(16, 2)
    y_correct = torch.randint(0, 2, (16, 1)).float()

    with torch.no_grad():
        prediction = model(X)

    print("shape X :", tuple(X.shape))
    print("shape y :", tuple(y_correct.shape))
    print("shape output model :", tuple(prediction.shape))
    print("jumlah fitur input :", X.shape[1])
    print("jumlah output model:", prediction.shape[1])

    # --- Experiment Without Label Shape ---
    print("\n-- Mismatch  Shape (16,) --")
    y_wrong = torch.randint(0, 2, (16,)).float()  # Typo tanpa Shape
    print("shape y          :", tuple(y_wrong.shape))
    print("shape prediction :", tuple(prediction.shape))

    criterion = nn.BCEWithLogitsLoss()

    broadcast_result = prediction - y_wrong
    print("hasil broadcasting shape:", tuple(broadcast_result.shape))


    try:
        loss_wrong = criterion(prediction, y_wrong)
        print("UNEXPECTED: no error raised, loss:", loss_wrong.item())
    except ValueError as e:
        print("ValueError raised, as expected:")
        print(" ", e)

if __name__ == "__main__":
    demo_label_shape()
