"""Minimal end-to-end example: fit a small MLP with Hakuli.

Runs on CPU or GPU with no external dependencies beyond PyTorch.
"""

import torch
import torch.nn as nn

from hakuli import Hakuli


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_samples, dim = 4096, 64
    x = torch.randn(n_samples, dim, device=device)
    w_true = torch.randn(dim, dim, device=device) / dim**0.5
    y = x @ w_true + 0.01 * torch.randn(n_samples, dim, device=device)

    model = nn.Sequential(
        nn.Linear(dim, 256),
        nn.GELU(),
        nn.Linear(256, dim),
    ).to(device)

    low_rank = [p for p in model.parameters() if p.ndim == 2]
    regular = [p for p in model.parameters() if p.ndim != 2]

    optimizer = Hakuli(
        [
            {"params": low_rank, "rank": 64, "scale": 1.0, "update_proj_gap": 100},
            {"params": regular, "lr": 1e-3},
        ],
        lr=5e-4,
    )

    loss_fn = nn.MSELoss()
    for step in range(1000):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"step {step + 1:5d}  loss {loss.item():.6f}")


if __name__ == "__main__":
    main()
