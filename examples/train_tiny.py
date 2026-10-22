"""End-to-end Hakuli example on a small model (CPU-friendly).

Trains a tiny MLP on a synthetic regression task and prints the loss
trajectory. Run:

    python examples/train_tiny.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hakuli import Hakuli

RANK = 64
LR = 3e-3
SCALE = 4.0
STEPS = 300


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(64, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 32),
    )


def split_params(model: nn.Module, rank: int):
    lowrank, regular = [], []
    for p in model.parameters():
        if p.ndim == 2 and min(p.shape) > rank:
            lowrank.append(p)
        else:
            regular.append(p)
    return lowrank, regular


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)
    lowrank, regular = split_params(model, RANK)

    optimizer = Hakuli(
        [
            {"params": regular},
            {"params": lowrank, "rank": RANK, "scale": SCALE},
        ],
        lr=LR,
        update_proj_gap=50,
        foreach=True,
    )

    x = torch.randn(4096, 64, device=device)
    w_true = torch.randn(64, 32, device=device)
    y = torch.tanh(x @ w_true)

    for step in range(1, STEPS + 1):
        idx = torch.randint(0, x.shape[0], (128,), device=device)
        optimizer.zero_grad()
        loss = (model(x[idx]) - y[idx]).pow(2).mean()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:>4}  loss {loss.item():.5f}")


if __name__ == "__main__":
    main()
