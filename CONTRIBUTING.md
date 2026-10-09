# Contributing to Hakuli

Thanks for your interest in improving Hakuli. This document describes the
development workflow.

## Development setup

```bash
git clone https://github.com/xaedro-ai/hakuli.git
cd hakuli
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest -q
```

The suite runs on CPU and finishes in seconds. Please make sure it passes
before opening a pull request. New behaviour should come with tests; bug fixes
should include a regression test.

## Code style

- Target Python 3.9+ syntax (the CI matrix tests 3.9–3.12).
- Keep the optimizer dependency-light: PyTorch only, no optional imports.
- Public API changes must update the docstrings in `hakuli/optim.py`, the
  README configuration table, and the FAQ where relevant.
- Formatting follows PEP 8; keep lines under 100 characters.

## Pull requests

1. Fork the repository and create a topic branch.
2. Make focused commits with clear messages.
3. Ensure `pytest -q` passes.
4. Describe the motivation and the change in the pull request body. For
   algorithmic changes, include a short note on numerical impact (memory,
   step behaviour, or test evidence).

## Reporting issues

Use the repository issue tracker. For security-sensitive reports, follow
`SECURITY.md` instead of opening a public issue.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0 (see `LICENSE`).
