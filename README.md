# Hakuli

**Memory-efficient optimizer for full-parameter LLM training: no SVD, low-rank optimizer state, PyTorch-only.**

Hakuli is a memory-efficient optimizer for training and fine-tuning large
language models on a single consumer GPU. It performs full-parameter (not
adapter-based) updates while keeping optimizer memory close to SGD, by
approximating channel-wise gradient scaling with an auxiliary low-rank state
built from pure random projections. Hakuli performs **no SVD**, not even
periodically, which removes the periodic SVD stalls of GaLore-style
projection methods.

Hakuli is developed by XAEDRO and is used in the company's internal LLM
training pipeline.

Hakuli is an independent, clean-room implementation of the APOLLO optimization
method (Zhu et al., MLSys 2025). The algorithm follows the published
specification; no source code from the reference implementation was used.
Hakuli is not affiliated with, or endorsed by, the APOLLO authors. See
[`NOTICE`](NOTICE) for academic attributions.

[![CI](https://github.com/xaedro-ai/hakuli/actions/workflows/ci.yml/badge.svg)](https://github.com/xaedro-ai/hakuli/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hakuli.svg)](https://pypi.org/project/hakuli/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)

---

## Why Hakuli

| | AdamW | GaLore | **Hakuli** |
|---|---|---|---|
| Full-parameter training | Yes | Yes | Yes |
| Optimizer memory (7B, rank 256) | ≈54 GB (LLaMA-7B) | ~9 GB | **~2.4 GB** |
| Periodic SVD | None | Every `gap` steps | **None** |
| Dependency footprint | torch | galore-torch | **torch only** |

- **No SVD anywhere.** Projection matrices are Gaussian random and resampled on
  a fixed cadence. There is no `O(mn²)` decomposition, no rebuild stalls.
- **Low-rank auxiliary state.** Moments are tracked in a `rank × n` subspace;
  the update is scaled channel-wise from that state and applied to the full
  gradient. The weights remain full-rank.
- **Norm-growth limiter.** A relative update-norm limiter (`nl_threshold`,
  default `1.01`) caps update-norm growth between consecutive steps,
  converting abrupt gradient spikes into smooth increases.
- **Minimal API.** A single `torch.optim.Optimizer` subclass. Works with
  plain PyTorch training loops, gradient accumulation, and per-layer setups.

## What is new in 2.x

**2.1** adds experimental update-rule options, off by default and measured
(see [Experimental options](#experimental-options-21) and
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)).

**2.0** is an execution and tooling revision; the optimizer mathematics is
unchanged. Validated with the test suite and the measured benchmarks in this
repository (RTX 3090, PyTorch 2.11):

- **Synchronization-free step.** 1.x called `.item()` once per parameter per
  step (a host round-trip that stalls the device). 2.0 keeps the norm-limiter
  state on the device: `step()` never synchronizes, verified by a test that
  forbids `Tensor.item` / `__bool__` inside the step.
- **Fused parameter update.** The update is applied as a single
  `addcmul_` pass (`w <- w - eta * g * s`) instead of materializing and
  rescaling a full-size update tensor four times.
- **Matmul-ready projection cache.** The projection is stored transposed
  once at resample time and in the gradient's dtype; no per-step casts of
  `rank x max(m, n)` buffers.
- **`foreach=True`**: batched `torch._foreach_*` path for bias/norm
  parameters (identical results to the per-parameter path, verified).
- **`limiter_norm="projected"`**: evaluates the norm-growth limiter in the
  rank-dimensional subspace (Johnson-Lindenstrauss estimate), removing the
  last full-size temporary from the step.

Measured, optimizer-only microbenchmark (8192x8192 matrix, rank 256, bf16):

| | v1.0 | v2.0 | v2.0 projected limiter |
|---|---|---|---|
| ms/step | 2.53 | 1.99 | **1.46** |
| peak transient VRAM | 272 MB | 144 MB | **24 MB** |

Two caveats, measured and documented honestly:

- With *many small parameters* (bias/norm tensors) the sync-free path can be
  slower than 1.x in CPU-bound microbenchmarks; set `foreach=True` there
  (measured 29.6 ms vs 33.8 ms for 1.x on the mixed LLM workload).
- `transport_moments` (re-expressing moments in the new basis at resample)
  was implemented, measured consistently **worse** than letting the
  old-basis content decay (+0.08 nats at gap=50, three seeds), and is kept
  off by default as a documented negative result.

## Installation

Hakuli is dependency-light (PyTorch only). Install from PyPI:

```bash
pip install hakuli
```

Or from source:

```bash
git clone https://github.com/xaedro-ai/hakuli.git
cd hakuli
pip install -e .
```

Requires Python 3.9+ and PyTorch 2.1+. CUDA is recommended but not required.

## Quick start

```python
import torch
import torch.nn as nn
from hakuli import Hakuli

model = nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512)).cuda()

# Hakuli routes 2-D weights through the low-rank path; parameters whose
# smallest dimension is not larger than `rank` (including 1-D parameters)
# automatically use its Adam-style fallback.
lowrank, regular = [], []
for name, p in model.named_parameters():
    (lowrank if p.ndim == 2 else regular).append(p)

optimizer = Hakuli(
    [
        {"params": regular},
        {"params": lowrank, "rank": 256, "scale": 1.0,
         "scale_type": "channel", "update_proj_gap": 200},
    ],
    lr=1e-5,
)

for step, (x, y) in enumerate(loader):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

A complete runnable example (small model, CPU) is provided in
[`examples/train_tiny.py`](examples/train_tiny.py).

### Per-parameter (layer-wise) setups

Hakuli is a standard optimizer and can be instantiated per parameter. This is
the recommended pattern when gradients must be released eagerly:

```python
optimizers = {id(p): Hakuli([p], lr=1e-5, rank=256) for p in lowrank_params}

def hook(p):
    if p.grad is None:
        return
    opt = optimizers[id(p)]
    opt.step()
    opt.zero_grad(set_to_none=True)

for p in lowrank_params:
    p.register_post_accumulate_grad_hook(hook)
```

## Configuration

| Argument | Default | Description |
|---|---|---|
| `lr` | `1e-5` | Learning rate. The APOLLO-style scale-corrected step size is applied internally; no external scheduler is required. |
| `rank` | `256` | Rank of the auxiliary random subspace. 256 is a good default for 1B–7B models. |
| `scale` | `1.0` | Global multiplier on the structured update (applied under a square root). Tune together with `rank`. |
| `scale_type` | `"channel"` | `"channel"` (APOLLO-style) or `"tensor"` (extreme low-memory). |
| `update_proj_gap` | `200` | Steps between random-projection resamples. |
| `betas` | `(0.9, 0.999)` | Moment decay coefficients. |
| `eps` | `1e-8` | Numerical stability term. |
| `weight_decay` | `0.0` | Decoupled weight decay. |
| `nl_threshold` | `1.01` | Relative cap on update-norm growth between steps. |
| `limiter_norm` | `"exact"` | `"projected"` estimates the update norm in the subspace (no full-size temporary). |
| `foreach` | `False` | Batch the bias/norm path with `torch._foreach_*`. Use it with many small parameters. |
| `transport_moments` | `False` | Experimental; measured worse than decay. Kept as a documented negative result. |

## Experimental options (2.1)

All off by default, measured on the tiny GPT and on LLaMA-7B short runs
([`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)):

| Option | Default | What it does | Measured |
|---|---|---|---|
| `anti_windup=True` | `False` | Corrects the moment windup caused by the limiter while it is saturated | Neutral in smooth regimes; **−0.14 nats** in the spike regime |
| `denom="update"` | `"grad"` | AdaDelta-style denominator over applied updates | Kills `scale` sensitivity (−95%) at a quality cost of +0.48 nats |
| `direction="atan2"` | `"adam"` | `atan2(m, sqrt(v))` update direction | Neutral / slightly worse |
| `norm_estimator="chi"` | `"sqrt_rank"` | Chi-distribution correction for the projected limiter | Neutral at rank 64 (matters at rank ≤ 8) |
| `nl_schedule="record"` | `"fixed"` | Limiter threshold from recorded update norms | Neutral; no gain in the spike regime |
| `scale_type="two_sided"` | `"channel"` | Channel scale applied on both sides | Neutral / slightly worse |

`anti_windup` is the one option kept as a candidate for the default rule;
the others are regime-specific insurance or documented negative results.

## What the optimizer actually does (measured invariants)

These were verified numerically against the implementation:

1. **The norm limiter is a soft running minimum.** With the default
   `nl_threshold=1.01`, the limiter state follows exactly
   `prev_{t+1} = min(un_t, 1.01 * prev_t)` (max deviation 8e-8 over 40
   steps). It engages in 18% of steps at rank 64 and 52% at rank 8 with
   i.i.d. gradients; on a fixed gradient with resampling every 25 steps it
   engages in 61% of steps, absorbing basis-change transients.
2. **The update is self-normalized.** `||delta_w|| ≈ lr * sqrt(cols) * sqrt(scale)`,
   independent of gradient magnitude (a 1000x gradient spike produces the
   same update norm) and constant across steps. The `eta_t` correction
   exactly cancels the Adam-style scale growth.
3. **`lr` and `scale` are redundant for the projected path**: only
   `lr * sqrt(scale)` matters (verified: two configurations with equal
   product produce bit-identical trajectories).
4. **The effective per-coordinate learning rate is
   `lr * sqrt(scale) * SNR / sqrt(rows)`** — it depends on each layer's
   shape. When tuning, compare `lr * sqrt(scale)` across layers; the
   `scale ≈ hidden / rank` heuristic of 1.x is only a starting point.

## Recommended usage notes

- Apply Hakuli to attention and MLP projection matrices plus embeddings; keep
  normalization layers and biases in the same optimizer (it routes them
  automatically) or in a conventional optimizer.
- Start from `lr` in the AdamW range for the same model, then calibrate with
  the effective-LR relation above. `scale` trades off with `rank`; the
  `scale ≈ hidden / rank` rule of thumb is only a starting point.
- With eager gradient release (per-parameter hooks), peak memory is
  dominated by activations, not optimizer state; combine with activation
  checkpointing for long-context training.
- Hakuli is orthogonal to precision schemes: it operates on whatever gradient
  dtype your model produces (`bf16` recommended).

## Limitations

- The optimizer targets fine-tuning-scale training. From-scratch
  pre-training over very long horizons is supported by the algorithm but
  **not yet validated at scale** by XAEDRO (the long-run behaviour of the
  limiter ratchet and the effective-LR calibration are the open questions).
- Sparse gradients are not supported.
- `state_dict` follows the standard PyTorch convention of casting floating
  point state to the parameter dtype. **State dicts from 1.x are not
  directly loadable** (2.0 stores projections transposed, in matmul-ready
  layout).
- The projected limiter mode is an estimator (Johnson-Lindenstrauss); the
  exact mode remains the default.

## Documentation

- [`paper/hakuli-technical-report.pdf`](paper/hakuli-technical-report.pdf):
  compiled technical report (1.0).
- [`paper/`](paper/): report sources — LaTeX (`main.tex`, `references.bib`)
  and Markdown (`paper.md`).
- [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md): measured memory,
  throughput and feature-pack results.
- [`docs/design.md`](docs/design.md): algorithm, update rule, execution
  design, memory model.
- [`docs/faq.md`](docs/faq.md): batch sizes, rank selection, integrations.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The suite covers the update math against an fp64 reference, the foreach/single
equivalence, the sync-free property, state-dict round-trips, the documented
invariants, and end-to-end training behaviour.

## Citation

If you use Hakuli in academic work, please cite the underlying research that
inspired its design:

```bibtex
@inproceedings{zhu2025apollo,
  title     = {{APOLLO}: {SGD}-like Memory, {AdamW}-level Performance},
  author    = {Zhu, Hanqing and Zhang, Zhenyu and Cong, Wenyan and Liu, Xi and
               Park, Sem and Chandra, Vikas and Long, Bo and Pan, David Z. and
               Wang, Zhangyang and Lee, Jinwon},
  booktitle = {Conference on Machine Learning and Systems (MLSys)},
  year      = {2025}
}

@inproceedings{zhao2024galore,
  title     = {{GaLore}: Memory-Efficient {LLM} Training by Gradient Low-Rank Projection},
  author    = {Zhao, Jiawei and Zhang, Zhenyu and Chen, Beidi and Wang, Zhangyang and
               Anandkumar, Anima and Tian, Yuandong},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

"Hakuli" and "XAEDRO" are trademarks of XAEDRO. Hakuli is an independent
clean-room implementation; see the NOTICE file for academic attributions.
