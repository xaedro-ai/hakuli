"""Shared measurement harness for Hakuli benchmarks.

Provides model construction, dataset streaming, optimizer factories and a
training loop that records per-step metrics (loss, gradient norm, peak VRAM,
throughput), validation loss on held-out batches, and final run statistics.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch


GIB = 1024**3


@dataclass
class RunResult:
    method: str
    model: str
    dataset: str
    batch_size: int
    seq_len: int
    steps: int
    warmup_steps: int
    lr: float
    rank: Optional[int]
    scale: Optional[float]
    update_proj_gap: Optional[int]
    peak_vram_gib: float
    reserved_vram_gib: float
    tokens_per_sec: float
    final_loss: float
    final_val_loss: Optional[float]
    wall_time_s: float
    gpu: str
    torch_version: str
    extra: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def build_model(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_id.startswith("random:"):
        from transformers import LlamaConfig, LlamaForCausalLM

        _, _, layers, hidden = model_id.split(":")
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=int(hidden),
            intermediate_size=int(hidden) * 4,
            num_hidden_layers=int(layers),
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=4096,
        )
        model = LlamaForCausalLM(config)
        tokenizer = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer


def synthetic_batches(
    batch_size: int, seq_len: int, vocab_size: int, steps: int, device: str
) -> Iterator[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(steps):
        ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), generator=generator
        )
        yield ids.to(device)


def packed_batches(
    tokenizer,
    dataset: str,
    batch_size: int,
    seq_len: int,
    steps: int,
    device: str,
) -> Iterator[torch.Tensor]:
    from datasets import load_dataset

    if dataset == "c4":
        stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
        stream = (example["text"] for example in stream)
    elif dataset == "smoltalk":
        raw = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)

        def render(example):
            return tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )

        stream = (render(example) for example in raw)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    buffer: list[int] = []
    eos = tokenizer.eos_token_id
    produced = 0
    for text in stream:
        buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buffer.append(eos)
        while len(buffer) >= batch_size * seq_len:
            chunk = buffer[: batch_size * seq_len]
            buffer = buffer[batch_size * seq_len :]
            yield torch.tensor(chunk, dtype=torch.long, device=device).view(
                batch_size, seq_len
            )
            produced += 1
            if produced >= steps:
                return
    raise RuntimeError("Dataset stream exhausted before reaching the step budget")


def _per_param_low_rank(parameters, rank: int):
    low_rank = [
        p for p in parameters if p.requires_grad and p.ndim == 2 and min(p.shape) > rank
    ]
    low_rank_ids = {id(p) for p in low_rank}
    regular = [p for p in parameters if p.requires_grad and id(p) not in low_rank_ids]
    return low_rank, regular


def make_optimizers(
    method: str,
    model,
    lr: float,
    rank: int,
    scale: float,
    gap: int,
    scale_front: bool = False,
):
    """Return (per_parameter_optimizers, shared_optimizer).

    Low-rank methods use one optimizer per parameter (stepped from gradient
    hooks during the backward pass); the shared optimizer covers all remaining
    parameters and is stepped once per iteration.
    """
    parameters = list(model.parameters())
    low_rank, regular = _per_param_low_rank(parameters, rank)
    shared = None
    per_param: list[tuple[torch.nn.Parameter, torch.optim.Optimizer]] = []

    if method == "adamw":
        shared = torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.95))
        return per_param, shared

    if method == "hakuli":
        from hakuli import Hakuli

        for p in low_rank:
            per_param.append(
                (
                    p,
                    Hakuli(
                        [p],
                        lr=lr,
                        rank=rank,
                        scale=scale,
                        update_proj_gap=gap,
                        scale_type="channel",
                    ),
                )
            )
    elif method == "galore":
        from galore_torch import GaLoreAdamW8bit

        for p in low_rank:
            per_param.append(
                (
                    p,
                    GaLoreAdamW8bit(
                        [{"params": [p], "rank": rank, "update_proj_gap": gap,
                          "scale": scale, "proj_type": "std"}],
                        lr=lr,
                    ),
                )
            )
    elif method == "apollo":
        from apollo_torch import APOLLOAdamW

        for p in low_rank:
            per_param.append(
                (
                    p,
                    APOLLOAdamW(
                        [{"params": [p], "rank": rank, "update_proj_gap": gap,
                          "scale": scale, "scale_type": "channel",
                          "proj": "random", "proj_type": "std",
                          "scale_front": scale_front}],
                        lr=lr,
                    ),
                )
            )
    elif method == "qgalore":
        from q_galore_torch import QGaLoreAdamW8bit

        for p in low_rank:
            per_param.append(
                (
                    p,
                    QGaLoreAdamW8bit(
                        [{"params": [p], "rank": rank, "update_proj_gap": gap,
                          "scale": scale, "proj_type": "std",
                          "proj_quant": True, "weight_quant": True,
                          "stochastic_round": True}],
                        lr=lr,
                    ),
                )
            )
    else:
        raise ValueError(f"Unknown method: {method}")

    if regular:
        shared = torch.optim.AdamW(regular, lr=lr, betas=(0.9, 0.95))
    return per_param, shared


def attach_hooks(per_param, stats: Optional[dict] = None):
    lookup = {id(p): opt for p, opt in per_param}

    def hook(param):
        opt = lookup.get(id(param))
        if opt is None or param.grad is None:
            return
        if stats is not None:
            stats["grad_norm_sq"] += float(param.grad.detach().norm()) ** 2
        opt.step()
        opt.zero_grad(set_to_none=True)

    for p, _ in per_param:
        p.register_post_accumulate_grad_hook(hook)


def evaluate(model, batches: list[torch.Tensor]) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for ids in batches:
            total += model(input_ids=ids, labels=ids).loss.item()
    if was_training:
        model.train()
    return total / len(batches)


def _resolve(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else repo_root / path


def run(config: dict, repo_root: Path) -> RunResult:
    from transformers import set_seed

    method = config["method"]
    model_id = config["model"]
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    steps = config["steps"]
    warmup_steps = config.get("warmup_steps", 5)
    lr = config["lr"]
    rank = config.get("rank", 256)
    scale = config.get("scale", 1.0)
    gap = config.get("update_proj_gap", 200)
    seed = config.get("seed", 0)
    eval_batches_n = config.get("eval_batches", 0)
    eval_every = config.get("eval_every", 0)
    print_every = config.get("print_every", 10)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)
    model, tokenizer = build_model(model_id, device)

    total_batches = steps + eval_batches_n
    if dataset == "synthetic":
        batches = synthetic_batches(
            batch_size, seq_len, model.config.vocab_size, total_batches, device
        )
    else:
        batches = packed_batches(
            tokenizer, dataset, batch_size, seq_len, total_batches, device
        )

    eval_batches = [next(batches) for _ in range(eval_batches_n)]

    per_param, shared = make_optimizers(
        method, model, lr, rank, scale, gap, config.get("scale_front", False)
    )
    stats = {"grad_norm_sq": 0.0}
    attach_hooks(per_param, stats)

    log_path = _resolve(
        config.get("log_jsonl", str(Path(config["output"]).with_suffix(".jsonl"))),
        repo_root,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")

    def log_record(record: dict) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    if eval_batches:
        initial_val = evaluate(model, eval_batches)
        log_record({"step": 0, "val_loss": round(initial_val, 4)})

    losses: list[float] = []
    steady_tokens = 0
    steady_time = 0.0
    window_start = time.perf_counter()
    window_tokens = 0
    start = time.perf_counter()

    model.train()
    for step in range(steps):
        if step == warmup_steps and device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if step < warmup_steps:
            steady_time = time.perf_counter()
        ids = next(batches)
        stats["grad_norm_sq"] = 0.0
        model.zero_grad(set_to_none=True)
        output = model(input_ids=ids, labels=ids)
        output.loss.backward()
        if shared is not None:
            if not per_param:
                for p in model.parameters():
                    if p.grad is not None:
                        stats["grad_norm_sq"] += float(p.grad.detach().norm()) ** 2
            shared.step()
        losses.append(output.loss.item())
        window_tokens += ids.numel()
        if step >= warmup_steps:
            steady_tokens += ids.numel()

        update_norm = None
        if method == "hakuli":
            update_norm = sum(
                opt.state[p].get("prev_norm", 0.0) for p, opt in per_param
            )

        should_log = (step + 1) % print_every == 0 or step == steps - 1
        should_eval = eval_every > 0 and (step + 1) % eval_every == 0
        if should_log or should_eval:
            now = time.perf_counter()
            window_secs = now - window_start
            record = {
                "step": step + 1,
                "loss": round(output.loss.item(), 4),
                "grad_norm": round(stats["grad_norm_sq"] ** 0.5, 4),
                "lr": lr,
                "tokens_per_sec_window": round(window_tokens / window_secs, 1)
                if window_secs > 0
                else 0.0,
                "peak_vram_gib": round(
                    torch.cuda.max_memory_allocated() / GIB, 2
                )
                if device == "cuda"
                else 0.0,
            }
            if update_norm is not None:
                record["update_norm"] = round(float(update_norm), 4)
            if should_eval:
                record["val_loss"] = round(evaluate(model, eval_batches), 4)
            log_record(record)
            print(
                f"[{method}] step {step + 1}/{steps} loss {record['loss']:.4f} "
                f"grad {record['grad_norm']:.2f} "
                f"vram {record['peak_vram_gib']:.2f}GiB "
                f"tok/s {record['tokens_per_sec_window']:.0f}",
                flush=True,
            )
            window_start = now
            window_tokens = 0

    if eval_batches:
        final_val = evaluate(model, eval_batches)
        log_record({"step": steps, "val_loss": round(final_val, 4)})
    else:
        final_val = None
    log_file.close()

    if device == "cuda":
        steady_time = time.perf_counter() - steady_time
        peak = torch.cuda.max_memory_allocated() / GIB
        reserved = torch.cuda.max_memory_reserved() / GIB
    else:
        peak = reserved = 0.0
    wall = time.perf_counter() - start

    tail = losses[-20:] if len(losses) >= 20 else losses
    low_rank_methods = ("hakuli", "galore", "qgalore", "apollo")

    return RunResult(
        method=method,
        model=model_id,
        dataset=dataset,
        batch_size=batch_size,
        seq_len=seq_len,
        steps=steps,
        warmup_steps=warmup_steps,
        lr=lr,
        rank=rank if method in low_rank_methods else None,
        scale=scale if method in low_rank_methods else None,
        update_proj_gap=gap if method in low_rank_methods else None,
        peak_vram_gib=round(peak, 2),
        reserved_vram_gib=round(reserved, 2),
        tokens_per_sec=round(steady_tokens / steady_time, 1) if steady_time > 0 else 0.0,
        final_loss=round(sum(tail) / len(tail), 4),
        final_val_loss=round(final_val, 4) if final_val is not None else None,
        wall_time_s=round(wall, 1),
        gpu=gpu_name(),
        torch_version=torch.__version__,
        extra={"config": config},
    )
