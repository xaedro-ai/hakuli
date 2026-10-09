"""Plot training and validation loss curves from benchmark JSONL logs.

Usage:
    python benchmarks/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).resolve().parent
RESULTS = BENCH_DIR / "results"


def load(path: Path):
    train, val = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "loss" in record:
            train.append((record["step"], record["loss"]))
        if "val_loss" in record:
            val.append((record["step"], record["val_loss"]))
    return train, val


def main() -> None:
    runs = [
        ("Hakuli", RESULTS / "memory_llama7b_hakuli.jsonl", "tab:blue"),
        ("APOLLO (official)", RESULTS / "memory_llama7b_apollo_official.jsonl", "tab:red"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, path, color in runs:
        if not path.exists():
            continue
        train, val = load(path)
        if train:
            ax.plot(
                [step for step, _ in train],
                [loss for _, loss in train],
                color=color,
                lw=1.2,
                label=f"{label} (train)",
            )
        if val:
            ax.plot(
                [step for step, _ in val],
                [loss for _, loss in val],
                color=color,
                ls="--",
                marker="o",
                ms=3,
                label=f"{label} (val)",
            )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("LLaMA-7B, batch 16, seq 256, RTX 3090")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output = RESULTS / "loss_curves.png"
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
