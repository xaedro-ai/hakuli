# Hakuli Design Notes

This document describes the algorithm implemented by `hakuli.optim.Hakuli`,
the execution design of version 2.x, the invariants that were measured
numerically, and how the method relates to prior academic work.

Hakuli is an independent, clean-room implementation written from published
algorithmic descriptions. See the repository `NOTICE` file for attributions.

## 1. Motivation

Full-parameter fine-tuning of modern language models is dominated by
optimizer memory: AdamW keeps two fp32 moments per trainable parameter
(about 8 bytes per parameter, ~54 GB for LLaMA-7B). Low-rank optimizers
reduce this by tracking moments in a compressed subspace, but SVD-based
projection methods (GaLore) pay a periodic decomposition cost and degrade at
small batch sizes. Hakuli removes the SVD entirely: the subspace is generated
by Gaussian random projections, cheap to draw and cheap to resample.

## 2. Update rule

For a 2-D parameter with gradient `G_t ∈ R^{m×n}` and `m ≥ n`:

```
R_t = P_t G_t                      (projected gradient, P_t ∈ R^{r×m})
M_t = β1·M_{t-1} + (1-β1)·R_t      (first moment, shape r×n)
V_t = β2·V_{t-1} + (1-β2)·R_t²     (second moment, shape r×n)
S_t = M_t / (sqrt(V_t) + ε)        (Adam-normalized direction)
s_t = ||S_t||_dim / ||R_t||_dim    (channel-wise scale)
U_t = sqrt(λ) · s_t ⊙ G_t          (full-rank structured update)
W_t = W_{t-1} - η_t·U_t,   η_t = lr·sqrt(1-β2^t)/(1-β1^t)
```

When `m < n` the projection is applied on the right (`R_t = G_t P_t`) and
channel norms are taken along rows. With `scale_type="tensor"` the vector
`s_t` is replaced by the scalar `||S_t||_F / ||R_t||_F`. Parameters that are
not 2-D, or whose smallest dimension does not exceed `rank`, use the
Adam-style direction `S_t` directly.

`P_t` is drawn from a standard Gaussian once and resampled every
`update_proj_gap` steps. The moments are carried across resamples; their
old-basis content decays exponentially with the moment half-lives (about
`1/(1-β1) ≈ 10` steps for the default betas).

## 3. Execution design (2.0)

- **Sync-free.** 1.x called `.item()` on the update norm once per parameter
  per step. 2.0 keeps `prev_norm` and all limiter arithmetic on the device.
  A test forbids `Tensor.item`, `Tensor.__bool__` and `Tensor.__float__`
  inside `step()`.
- **Fused update.** `scaling_out = s · sqrt(λ)/factor` is computed on the
  small scale vector, then the update is applied with one `addcmul_`:
  `w ← w − η·G·scaling_out`. No full-size update tensor is materialized in
  the projected path (with `limiter_norm="exact"`, one temporary is used to
  measure the exact update norm).
- **Projection cache.** The projection is stored in matmul-ready layout
  (`rank × m` for tall parameters, `n × rank` for wide ones) and in the
  gradient dtype, created once per resample. No per-step `.t()`/`.to()`
  copies.
- **foreach path.** With `foreach=True`, bias/norm parameters are batched
  through `torch._foreach_*` kernels; the per-parameter limiter and
  application remain scalar-loop but allocation-free.

## 4. Measured invariants

These were verified numerically against the implementation; they are
properties of the design worth knowing when tuning.

### 4.1 The norm limiter is a soft running minimum

With `nl_threshold = thr`, the limiter state satisfies exactly

```
prev_{t+1} = min(un_t, thr · prev_t)
```

(max relative deviation 8e-8 over 40 steps; `un_t` is the raw update norm).
Consequences:

- Upward jumps in the raw update norm are capped at `thr` per step;
  downward jumps pass freely. Under symmetric noise this is a rectifier that
  drags the update magnitude toward the lower envelope (a synthetic
  lognormal(0, 0.3) sequence settles at 0.48x its mean).
- Engagement is not rare: 52% of steps at rank 8, 18% at rank 64 with i.i.d.
  gradients (low rank means a noisier scale estimate), and 61% of steps with
  a fixed gradient resampled every 25 steps (basis-change transients).
- A zero-gradient step resets the baseline (`prev = 0`), which re-arms
  unconstrained growth on the next step (device-side `where`, no host
  branch).

### 4.2 The update is self-normalized

The channel scaling makes each update column roughly unit-norm, so

```
||Δw|| ≈ lr · sqrt(cols) · sqrt(scale)
```

independent of gradient magnitude (a 1000x spike produces the same update
norm, measured ratio 0.91) and constant across steps: the `η_t` correction
cancels the Adam-style direction-norm growth exactly (step-1 and step-60
norms equal to 3 decimals). Gradient clipping is therefore unnecessary for
magnitude; the limiter's real work is taming scale-estimate noise and
basis-change transients.

### 4.3 `lr` and `scale` are redundant in the projected path

Only the product `lr · sqrt(scale)` matters. Two configurations with equal
product produce bit-identical parameter trajectories (measured max deviation
0.0). The effective per-coordinate step is

```
lr_eff ≈ lr · sqrt(scale) · SNR / sqrt(rows)
```

so it depends on each layer's shape. When tuning across layers, compare
`lr · sqrt(scale)`; `scale ≈ hidden/rank` from 1.x is only a starting point.

### 4.4 Basis transport is a negative result

`transport_moments=True` re-expresses `M`/`V` in the new basis at resample
using `A = P_new P_oldᵀ / rank` (`V ← (A⊙A)V` as its diagonal
approximation). Measured consistently worse than decay: +0.084/+0.081/+0.080
nats at gap=50 and +0.007 at gap=200 over three seeds (tiny GPT). The
transported old-basis content is mostly
moment noise; decay washes it out, transport keeps it alive in a new noisy
basis. Default off.

## 5. Memory model

Per projected parameter of shape `m×n` (`r = rank`):

| Buffer | Shape | dtype | Bytes |
|---|---|---|---|
| Projection `P` | `r × max(m,n)` | gradient dtype | `2·r·max(m,n)` (bf16) |
| First moment `M` | `r × min(m,n)` | float32 | `4·r·min(m,n)` |
| Second moment `V` | `r × min(m,n)` | float32 | `4·r·min(m,n)` |

Worked example: LLaMA-7B, `r = 256` → total optimizer state ≈ 2.4 GB
versus ≈ 54 GB for AdamW fp32 moments.

## 6. Differences from prior work

| | GaLore (ICML'24) | APOLLO (MLSys'25) | Hakuli 2.0 |
|---|---|---|---|
| Projection | SVD of gradient, periodic | random, fixed | random, resampled every `gap` |
| SVD cost | `O(mn·min(m,n))` per rebuild | none | none |
| Moments | projected | projected (scaled) | projected |
| Update | projected ascent + projection back | full-gradient, channel-scaled | full-gradient, channel-scaled |
| Execution | syncs per parameter | (reference impl.) | sync-free, fused, foreach option |
| Norm-growth limiter | — | — | yes (soft running minimum, measured) |

No source code from any of these projects is used.

## 7. Hyperparameter guidance

- `rank`: 128–512. `rank = 256` is a good default for 1B–7B models. Lower
  ranks reduce memory but increase limiter engagement (see 4.1) and
  estimate noise.
- `scale`: compensate with `rank`; compare configurations through
  `lr · sqrt(scale)` (4.3). For 7B fine-tuning the product typically lands
  near `lr·sqrt(scale) ≈ 4e-5` with AdamW-range `lr`.
- `update_proj_gap`: 100–500 steps. Smaller gaps refresh the basis more
  often at negligible cost (resample draws a `rank × m` Gaussian).
- `lr`: start from the AdamW range for the same model and calibrate with
  (4.3).
- `nl_threshold`: `1.01` default. Increase to allow faster recovery when
  the loss is known to be stable; note the running-minimum behaviour (4.1).
- `foreach=True`: recommended when the model has many small parameters
  (bias/norm-heavy architectures, embeddings with separate norms).
- Experimental update-rule options (2.1, all off by default): `anti_windup`,
  `denom="update"`, `direction="atan2"`, `norm_estimator="chi"`,
  `nl_schedule="record"`, `scale_type="two_sided"`. Measured A/B results are
  in `benchmarks/RESULTS.md`; only `anti_windup` is kept as a candidate for
  the default rule (neutral smooth, -0.14 nats in the spike regime).

## 8. Limitations

- Targets fine-tuning-scale training; from-scratch pre-training is supported
  by the algorithm but not yet validated at scale (open questions: long-run
  ratchet behaviour, effective-LR calibration).
- Sparse gradients are not supported.
- `state_dict` follows the standard PyTorch convention of casting floating
  point state to the parameter dtype. 1.x state dicts are not directly
  loadable (projection layout changed).
- `limiter_norm="projected"` is a Johnson-Lindenstrauss estimator; the exact
  mode is the default.

## 9. References

1. J. Zhao, Z. Zhang, B. Chen, Z. Wang, A. Anandkumar, Y. Tian.
   *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.*
   ICML 2024. arXiv:2403.03507
2. H. Zhu, Z. Zhang, W. Cong, X. Liu, S. Park, V. Chandra, B. Long, D. Z. Pan,
   Z. Wang, J. Lee. *APOLLO: SGD-like Memory, AdamW-level Performance.*
   MLSys 2025. arXiv:2412.05270
3. X. Chen, K. Feng, C. Li, X. Lai, X. Yue, Y. Yuan, G. Wang.
   *Fira: Can We Achieve Full-Rank Training of LLMs Under Low-rank
   Constraint?* NeurIPS 2025. arXiv:2410.01623
