# Hakuli FAQ

## Is Hakuli only for fine-tuning, or does it train from scratch?

It is a general full-parameter optimizer: nothing in the algorithm assumes a
pre-trained initialization, and our experiments train tiny models from random
init. What differs is validation: 1.x was benchmarked on SFT, the underlying
method (APOLLO) has published from-scratch pre-training results, but XAEDRO
has **not yet validated long-horizon pre-training at scale**. The open
questions are the long-run behaviour of the norm limiter (it is a soft
running minimum; under sustained noise it can drag update magnitudes down)
and the effective-LR calibration (see below).

## How do I choose `lr` and `scale`?

They are redundant in the projected path: only `lr * sqrt(scale)` matters,
and the effective per-coordinate step is approximately
`lr * sqrt(scale) * SNR / sqrt(rows)`. Practical recipe:

1. Start with `lr` in the AdamW range for your model.
2. Pick `scale ≈ hidden_size / rank` as a starting point.
3. Compare configurations by the product `lr * sqrt(scale)`, not by each
   knob separately. Two configurations with the same product give identical
   trajectories.

## When should I set `foreach=True`?

When the model has many small parameters (biases, norms). The sync-free path
is optimized for large matrices; with thousands of tiny tensors the
per-parameter kernel launches dominate and batching them through
`torch._foreach_*` recovers the speed (and can beat 1.x). Results are
identical to the default path (verified in the test suite).

## What rank should I use?

128–512; 256 is a good default for 1B–7B. Two measured effects to keep in
mind: lower rank means a noisier channel-scale estimate (the norm limiter
engages more often: 52% of steps at rank 8 vs 18% at rank 64 with i.i.d.
gradients), and the estimator variance of `s_t` grows as `1/sqrt(rank)`.

## Does the optimizer clip gradients?

No. The channel scaling self-normalizes the update: `||Δw|| ≈ lr·sqrt(cols)·sqrt(scale)`
independent of gradient magnitude (measured). Gradient-magnitude spikes do
not produce update spikes. External gradient clipping is not needed for
magnitude stability.

## Can I load a 1.x checkpoint into 2.0?

Optimizer state dicts, no: 2.0 stores the projection transposed
(matmul-ready), so 1.x states fail to load. Model weights are unaffected.
Export/import via a fresh optimizer or migrate the projection manually.

## How do I integrate it with HuggingFace training loops?

Treat it as any `torch.optim.Optimizer`: build parameter groups (2-D
matrices with `min(shape) > rank` in the low-rank group, the rest in the
default group) and pass the optimizer to your loop or `Trainer` (via
`optimizers=(optimizer, scheduler)`). See `examples/train_tiny.py`.

## Can Hakuli be used for inference?

No. It is a training optimizer; inference does not use optimizers at all.
Models trained with Hakuli have standard weights and run in any inference
engine.

## Why is `transport_moments` in the code if it does not work?

Because the negative result is worth preserving: it was implemented, tested
for correctness and measured consistently worse than the decay behaviour
(three seeds, two gap settings). Keeping it documented prevents the same
idea from being re-invented and re-measured later.
