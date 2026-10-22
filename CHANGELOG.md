# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-10-17

### Added

- Experimental update-rule options in `Hakuli`, all off by default and A/B
  measured on the tiny GPT and on LLaMA-7B short runs (see
  `benchmarks/RESULTS.md`):
  - `anti_windup=True`: corrects the moment windup caused by the norm
    limiter while it is saturated. Neutral in smooth regimes; **-0.14 nats**
    in the spike-injection regime. The one option kept as a candidate.
  - `denom="update"`: AdaDelta-style denominator over applied updates.
    Removes nearly all `scale` sensitivity (-95% measured) at a quality
    cost of +0.48 nats; needs a magnitude-preserving redesign.
  - `direction="atan2"`: `atan2(m, sqrt(v))` update direction.
  - `norm_estimator="chi"`: chi-distribution correction for the projected
    limiter (matters at rank <= 8; neutral at rank 64).
  - `nl_schedule="record"`: limiter threshold from recorded update norms.
  - `scale_type="two_sided"`: channel scale applied on both sides.

## [2.0.0] - 2026-10-11

### Added

- `limiter_norm="projected"`: the norm-growth limiter can be evaluated in the
  projection subspace (`||U|| ≈ sqrt(scale) * ||s * R|| / sqrt(rank)`),
  removing the last full-size temporary from the step.
- `foreach=True`: batched `torch._foreach_*` execution for the bias/norm path;
  produces identical results to the per-parameter path (tested) and is faster
  when the model has many small parameters.
- `transport_moments` (experimental, default off): re-expresses the stored
  moments in the new basis at every resample using the rank x rank overlap
  between consecutive projections. **Measured negative result** (consistently
  +0.08 nats at gap=50 and +0.007 at gap=200 over three seeds, tiny GPT), kept
  as a documented option for further research.
- Test suite expanded, including fp64 reference equivalence of the fused
  step, a test that forbids host synchronization inside `step()`, the
  foreach/single equivalence and the documented invariants.
- `benchmarks/RESULTS.md`: measured microbenchmark and parity results.

### Changed

- `step()` is synchronization-free: the norm-limiter state lives on the device
  (`prev_norm` is a device scalar); no `.item()`/`float()`/truth-value
  round-trips to the host anywhere in the update path.
- The parameter update is applied as a single fused `addcmul_` pass instead of
  materializing and rescaling a full-size update tensor several times.
- Projections are stored in matmul-ready layout (transposed once at resample
  time) and in the gradient's own dtype; no per-step casts.
- Zero-gradient steps restart the limiter baseline via a device-side `where`
  (no host branch, same semantics as 1.x).
- `state_dict` layout changed (see below).

### Compatibility

- State dicts saved with 1.x are **not directly loadable** into 2.0: the
  projection tensor is stored transposed (matmul-ready). Export/import via a
  fresh optimizer or migrate the projection manually.
- The public constructor arguments of 1.x are unchanged; 2.0 only adds
  optional arguments.

### Negative results (documented, measured)

- `transport_moments=True` is consistently worse than the exponential decay of
  old-basis moment content used by 1.x. The transported content is mostly
  moment noise; rotating it keeps it alive instead of washing it out.
- With many small parameters, the sync-free path can lose to 1.x in
  CPU-bound microbenchmarks unless `foreach=True` is set.

## [1.0.0] - 2026-10-10

Initial public release of the Hakuli optimizer (see the 1.x repository
history for details).
