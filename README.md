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

## Recommended usage notes

- Apply Hakuli to attention and MLP projection matrices plus embeddings; keep
  normalization layers and biases in a conventional optimizer.
- `rank` 256 with `scale` ~14 (`hidden_size / rank` rule of thumb) is a good
  starting point for 7B-scale fine-tuning.
- With eager gradient release (per-parameter hooks), peak memory is dominated
  by activations, not optimizer state; combine with activation checkpointing
  for long-context training.
- Hakuli is orthogonal to precision schemes: it operates on whatever gradient
  dtype your model produces (`bf16` recommended).

## Documentation

- [`paper/hakuli-technical-report.pdf`](paper/hakuli-technical-report.pdf):
  compiled technical report.
- [`paper/`](paper/): report sources — LaTeX (`main.tex`, `references.bib`)
  and Markdown (`paper.md`).
- [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md): measured memory and
  throughput results.
- [`docs/design.md`](docs/design.md): algorithm, update rule, memory model.
- [`docs/faq.md`](docs/faq.md): batch sizes, rank selection, integrations.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

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
