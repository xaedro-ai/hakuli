# Benchmark results

All numbers measured in this repository. Environment: **NVIDIA GeForce RTX
3090 (24 GB), PyTorch 2.11.0+cu128, Python 3.14**, unless stated otherwise.
The benchmark runs are reproducible from the configs in
`benchmarks/configs/`; the optimizer microbenchmark is
`benchmarks/microbench.py`.

## Optimizer-only microbenchmark (bf16)

`python benchmarks/microbench.py --device cuda --dtype bfloat16 --steps 30`

Fixed gradients, no resampling; measures the isolated optimizer step.

| Workload | Impl | ms/step | peak transient | state |
|---|---|---|---|---|
| single 8192x8192, rank 256 | v1.0 | 2.53 | 272 MB | 20 MB |
| | **v2.0** | **1.99** | 144 MB | 20 MB |
| | **v2.0 projected limiter** | **1.46** | **24 MB** | 20 MB |
| mixed LLM (24 mats + 200 vectors) | v1.0 | 33.8 | 48 MB | 143 MB |
| | v2.0 | 60.8 | 26 MB | 143 MB |
| | v2.0 `foreach=True` | **29.6** | 26 MB | 143 MB |
| | v2.0 projected limiter | 42.7 | 6 MB | 143 MB |
| many small params (2100 tensors) | v1.0 | **200** | 0 MB | 7 MB |
| | v2.0 | 351 | 0 MB | 7 MB |
| | v2.0 `foreach=True` | 320 | 8 MB | 7 MB |

Honest reading: 2.0 wins clearly on large matrices (fused update, no syncs,
no per-step casts). With many small parameters the per-parameter device-side
limiter machinery costs more kernel launches than 1.x's host-side floats in
this CPU-bound microbenchmark; `foreach=True` recovers most of it. In real
LLM training the optimizer is a small fraction of step time and the syncs
removed by 2.0 matter more; treat this table as an upper bound on the
optimizer's share.

## Training parity (tiny GPT, AdamW-class task)

Tiny GPT, 400 steps, rank 128, gap 50, lr 3e-2, scale 8, identical
init/data.

| Arm | eval loss | jitter at resample |
|---|---|---|
| v1.0 | 5.0268 | 0.00411 |
| v2.0 (no transport) | 5.0261 | 0.00411 |
| v2.0 (transport) | 5.1091 | 0.00435 |

v2.0 reproduces 1.x. `transport_moments` is a measured negative result
(+0.08 nats, consistent over three seeds at gap=50; +0.007 at gap=200).

## Documented invariants

- Limiter state == `min(un_t, 1.01 * prev_t)`: max deviation 7.8e-8.
- Update norm `||Δw|| = 0.0167` for gradient scales 1e-3, 1, 1e3
  (`lr·sqrt(cols) = 0.0160`); constant from step 1 to 60.
- `(lr=1e-3, scale=4)` vs `(lr=5e-4, scale=16)`: max difference 0.0.
- Limiter engagement: 52% (rank 8), 18% (rank 64), 61% (resample every 25).

## Research feature pack (experimental options, tiny GPT)

Tiny GPT A/B, 400 steps, rank 64, lr 3e-2, scale 8, seed 0.

Smooth regime (no spikes; limiter barely engages):

| Arm | eval | note |
|---|---|---|
| baseline | 5.0541 | — |
| adadelta (`denom="update"`) | 5.5359 | scale sensitivity killed (0.006 vs 0.134) but +0.48 nats quality cost |
| atan2 | 5.0893 | neutral/slightly worse |
| anti-windup | 5.0537 | neutral (no saturation to correct) |
| chi estimator | 5.0540 | neutral at rank 64 (matters at rank <= 8) |
| record schedule | 5.0554 | neutral |
| two-sided | 5.0628 | neutral/slightly worse |

Spike-injection regime (loss x100 every 40 steps; the real failure mode):

| Arm | eval | post-spike mean |
|---|---|---|
| baseline | 5.2862 | 5.3576 |
| **anti-windup** | **5.1473 (-0.14)** | **5.2969 (-0.06)** |
| record | 5.2960 | 5.3619 |
| both | 5.1512 | 5.2985 |

Verdict: **anti-windup is the keeper** — it corrects the moment windup caused by
the limiter and pays off exactly in the spike regime (-0.14 nats here).
Adadelta-denominator achieves its design goal (removes the scale hyperparameter
sensitivity) but needs a magnitude-preserving redesign before it is a
drop-in; the remaining options are regime-specific insurance.

## Anti-windup on LLaMA-7B (short spike test)

`benchmarks/run_benchmark.py --config benchmarks/configs/short_llama7b_*.json`
— 150 steps (vs 1000 of the full protocol), loss x50 every 25 steps,
identical seeds/data/hyperparameters.

| Metric | Baseline | Anti-windup |
|---|---|---|
| final val loss | 2.2275 | 2.2269 |
| final train loss | 2.1754 | 2.1749 |
| peak VRAM | 20.51 GiB | 20.51 GiB |
| tokens/s | 1057.7 | 1047.0 (-1%) |
| update norm after spike (step 50) | 7358 | 9331 (+27%) |
| update norm (step 100) | 2848 | 5205 (+83%) |

Reading: the mechanism is live (post-spike update norms are 27-83% larger
because the moments no longer integrate clipped steps) and safe on the real
model (neutral loss, identical VRAM, -1% throughput from the per-step
correction, which could be gated on saturation). 150 steps at lr 1e-5 barely
move the loss (2.2301 -> 2.2275 total), so this run cannot measure a quality
benefit; the +0.14 nats gain was measured in the tiny-GPT spike regime
(above). **Pending: LLaMA-7B spike-regime validation with 400 steps, lr 1e-4,
spikes every 20 (~50 min for both arms).**
