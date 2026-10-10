# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-10

### Added

- Initial public release of the Hakuli optimizer.
- `Hakuli` optimizer with:
  - Gaussian random low-rank projection (no SVD at any step),
  - Adam-style moments tracked in the projected subspace,
  - channel-wise (`"channel"`) and tensor-wise (`"tensor"`) update scaling,
  - periodic projection resampling (`update_proj_gap`),
  - relative norm-growth limiter (`nl_threshold`),
  - Adam-style fallback for non-2-D and small parameters,
  - decoupled weight decay and standard `state_dict` support.
- Unit and integration test suite.
- End-to-end example (`examples/train_tiny.py`).
- Documentation: design notes, FAQ, contribution and security policies.
- CI workflow running the test suite on Python 3.9/3.11/3.12.
