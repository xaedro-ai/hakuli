# FAQ

### How is Hakuli different from GaLore?

GaLore projects the gradient with the top singular vectors from a periodic SVD
and updates in the low-rank subspace. Hakuli uses a Gaussian random
projection, performs no SVD at any point, and applies the update to the full
gradient. Random projection removes the `O(mn·min(m,n))` rebuild stalls and is
stable at batch size 1.

### Is this AdamW? Is this SGD?

Neither exactly. The moment estimates and the update shape are Adam-like, but
the step is applied to the full gradient with a scale-corrected SGD-style
learning rate (`η_t = lr·sqrt(1-β2^t)/(1-β1^t)`), following the APOLLO
formulation. In practice you tune it like AdamW.

### What batch sizes are supported?

Hakuli is designed for small-batch, single-GPU training (batch 1–8) and works
at any batch size. Unlike SVD-based methods, the projection quality does not
degrade as the batch shrinks.

### Which parameters should I route to Hakuli?

2-D weight matrices (attention/MLP projections, embeddings) benefit the most.
Parameters whose smallest dimension is not larger than `rank`, and all 1-D
parameters (biases, layer norms), automatically use an Adam-style path, so you
can pass `model.parameters()` directly to a single Hakuli instance.

### How much memory does it save?

Per projected parameter, Hakuli stores `2·r·max(m,n)` (bf16 projection) plus
`8·r·min(m,n)` (two fp32 moments) bytes, versus `8·m·n` for AdamW. For a 7B
model at `rank=256`, optimizer state drops from ≈ 61 GB to ≈ 2.4 GB. See
[`design.md`](design.md) for the full derivation.

### Does it support gradient accumulation and mixed precision?

Yes. Accumulate gradients as usual and call `optimizer.step()` once per
effective batch. Gradients may be bf16/fp16/fp32; moments are kept in fp32.

### How do I checkpoint and resume?

Use the standard `state_dict()` / `load_state_dict()` API. The projection,
moments and limiter state are all restored. As with all PyTorch optimizers,
floating-point state is cast to the parameter dtype when serialized; the
bfloat16 projection values are preserved exactly.

### Can I use it with Hugging Face Trainer?

Yes — pass a Hakuli instance via the `optimizers=(optimizer, scheduler)`
argument. Build it over `model.parameters()`; the fallback path handles non-2-D
parameters.

### Why "no SVD" — is that not less accurate?

The channel scale only needs norm estimates, and both sides of the ratio are
computed in the same random subspace, so first-order distortion cancels.
Periodic resampling avoids basis overfitting. Accuracy in practice matches
SVD-based projection while being cheaper and more stable.

### Is sparse gradient training supported?

No. Sparse gradients raise `RuntimeError`. Dense training covers the intended
use cases (transformers, MLPs, embeddings).

### What is the license and provenance?

Apache License 2.0. Hakuli is an independent clean-room implementation of
published research ideas; see `NOTICE` for academic attributions.
