"""Deep Learning Session 4 — Framework Debugging Exercise."""

import torch
from torch import nn


def broken_training_step(model, optimizer, criterion, X, y):
    """Intentionally broken example for classroom code review."""
    model.eval()  # BUG: training should normally use model.train().
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def corrected_training_step(model, optimizer, criterion, X, y):
    """Corrected version with the training mode explicitly enabled."""
    model.train()
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def inspect_batch(model, X, y):
    """Print the checks students should perform before a long training run."""
    print("input shape:", tuple(X.shape), "dtype:", X.dtype, "device:", X.device)
    print("label shape:", tuple(y.shape), "dtype:", y.dtype, "device:", y.device)
    print("model device:", next(model.parameters()).device)
    with torch.no_grad():
        prediction = model(X)
    print("prediction shape:", tuple(prediction.shape), "dtype:", prediction.dtype)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    X = torch.randn(16, 2)
    y = torch.randint(0, 2, (16, 1)).float()
    inspect_batch(model, X, y)
    print("broken loss:", broken_training_step(model, optimizer, criterion, X, y))
    print("corrected loss:", corrected_training_step(model, optimizer, criterion, X, y))
