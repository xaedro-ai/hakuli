"""Optimizer-only microbenchmark: hakuli 1.x vs 2.0 variants.

Measures per-step wall time, peak transient GPU memory and optimizer state
size on synthetic parameter sets with realistic LLM shapes. Gradients are
held fixed so the measurement isolates the optimizer step.

Usage:
    python benchmarks/microbench.py --v1-dir "..\\hakuli" --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_optimizer_class(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Hakuli


def workload_shapes(name: str) -> Dict[str, List[tuple]]:
    if name == "single-big":
        return {"projected": [(8192, 8192)]}
    if name == "mixed-llama":
        return {
            "projected": [(2048, 2048)] * 12 + [(5632, 2048)] * 12,
            "fallback": [(2048,)] * 200,
        }
    if name == "many-fallback":
        return {
            "fallback": [(768,)] * 1500 + [(32, 32)] * 500 + [(2048,)] * 100
        }
    raise ValueError(f"unknown workload: {name}")


def build_params(shapes: Dict[str, List[tuple]], device, dtype):
    params, groups = [], []
    for shape_list in shapes.values():
        group = []
        for shape in shape_list:
            p = torch.nn.Parameter(
                torch.randn(*shape, device=device, dtype=dtype)
            )
            p.grad = torch.randn_like(p)
            params.append(p)
            group.append(p)
        groups.append({"params": group})
    return params, groups


def state_bytes(opt) -> int:
    total = 0
    for st in opt.state.values():
        for value in st.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def bench_one(name, opt, params, steps, warmup, device):
    for _ in range(warmup):
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
    else:
        peak = 0
    dt = (time.perf_counter() - t0) / steps
    return {
        "workload": name,
        "ms_per_step": dt * 1e3,
        "steps_per_s": 1.0 / dt,
        "peak_transient_mb": peak / 2**20,
        "state_mb": state_bytes(opt) / 2**20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir", type=str, default=str(ROOT.parent / "hakuli"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    device = args.device
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if device == "cpu" and dtype == torch.bfloat16:
        print("bf16 on CPU is not benchmarked; using float32")
        dtype = torch.float32

    HakuliV2 = load_optimizer_class(ROOT / "hakuli" / "optim.py", "hakuli_v2_bench")
    v1_path = Path(args.v1_dir) / "hakuli" / "optim.py"
    HakuliV1 = load_optimizer_class(v1_path, "hakuli_v1_bench") if v1_path.exists() else None

    variants = []
    if HakuliV1 is not None:
        variants.append(("v1.0", HakuliV1, {}))
    variants += [
        ("v2.0", HakuliV2, {}),
        ("v2.0 projected-limiter", HakuliV2, {"limiter_norm": "projected"}),
        ("v2.0 foreach", HakuliV2, {"foreach": True}),
    ]

    results = []
    header = f"{'workload':<16}{'impl':<24}{'ms/step':>10}{'steps/s':>10}{'peak MB':>10}{'state MB':>10}"
    print(header)
    print("-" * len(header))
    for wname in ("single-big", "mixed-llama", "many-fallback"):
        shapes = workload_shapes(wname)
        for impl_name, cls, kw in variants:
            torch.manual_seed(0)
            params, groups = build_params(shapes, device, dtype)
            opt = cls(groups, lr=1e-5, rank=args.rank, update_proj_gap=10**9, **kw)
            row = bench_one(wname, opt, params, args.steps, args.warmup, device)
            row["impl"] = impl_name
            results.append(row)
            print(
                f"{wname:<16}{impl_name:<24}{row['ms_per_step']:>10.3f}"
                f"{row['steps_per_s']:>10.1f}{row['peak_transient_mb']:>10.1f}"
                f"{row['state_mb']:>10.1f}"
            )

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
