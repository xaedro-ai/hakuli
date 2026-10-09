# Hakuli Benchmarks

Reproducible benchmarks for the Hakuli optimizer. Everything here is run by
`run_benchmark.py` with a JSON config; results land in `results/` as JSON and
`table.py` renders the comparison tables used in the paper and README.

## Hardware and protocol

- **Our measurements**: single NVIDIA RTX 3090 (24 GB), bf16, PyTorch SDPA
  attention, gradient checkpointing enabled, per-layer weight updates for
  low-rank methods (gradients released eagerly), packed token streams (no
  padding).
- **Published anchors**: measured by the respective authors on different
  hardware of the same 24 GB class or larger. They are cited for context; the
  protocol details differ (see each source). We re-run every method we can
  under one protocol for the controlled comparisons.

## Published anchors (LLaMA-7B)

| Method | GPU | Batch | Peak VRAM | Source |
|---|---|---|---|---|
| AdamW | 24 GB class | 16 | does not fit (fp32 states alone ≈ 61 GB) | GaLore paper |
| GaLore-8bit | RTX 4090 24GB | 16 | 22.8 GB | https://github.com/jiaweizzhao/GaLore |
| Q-GaLore-8bit | A6000 48GB | 16 | 15.26 GB | https://github.com/VITA-Group/Q-GaLore |
| APOLLO | A100 80GB | 1 | 11 GB | https://github.com/zhuhanqing/APOLLO |

The anchors are stored machine-readable in `results/published.json`.

## Running a benchmark

```bash
pip install -r benchmarks/requirements.txt
python benchmarks/run_benchmark.py --config benchmarks/configs/memory_llama7b_hakuli.json
```

Config fields: `method` (`hakuli`, `adamw`, `galore`, `qgalore`), `model`
(Hugging Face id, or `random:llama:<layers>:<hidden>` for smoke tests),
`dataset` (`c4`, `smoltalk`, `synthetic`), `batch_size`, `seq_len`, `lr`,
`steps`, `warmup_steps`, `rank`, `scale`, `update_proj_gap`, `seed`, `output`.

Measurements recorded per run: peak allocated and reserved VRAM, steady-state
tokens/second, loss (per-step and averaged over the final window), wall time,
GPU name, library versions, and the git revision of the repository.

## Rendering the tables

```bash
python benchmarks/table.py
```

## Notes on fairness

- All controlled runs share the model, dataset slice, token budget, sequence
  length, batch size, learning-rate schedule shape, and measurement code.
- Low-rank methods (Hakuli, GaLore, Q-GaLore) use per-layer weight updates so
  that gradient memory is released eagerly; AdamW uses a single optimizer.
- Memory numbers include model, gradients, activations and optimizer state as
  reported by the CUDA allocator (`torch.cuda.max_memory_allocated`).
- Different GPUs differ in available VRAM and allocator behavior; treat
  cross-hardware comparisons as indicative.
