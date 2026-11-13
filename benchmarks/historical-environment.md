# Historical benchmark environment

The October 2026 measurements in `RESULTS.md` and the technical report used:

| Component | Version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.5.1+cu124 |
| Transformers | 4.57.6 |
| Datasets | 5.0.1 |
| galore-torch | 1.0 |
| q-galore-torch | 1.0 |
| apollo-torch | 1.0.3 |

PyTorch came from the official CUDA 12.4 wheel index. Matplotlib was used
only for plotting, with no version recorded in the original manifest.

This is provenance for the recorded measurements, not an installation
manifest. The historical stack contains known vulnerabilities (see
`../SECURITY.md`). Use `requirements.txt` or the Dockerfile for new runs;
`requirements-repro.txt` forwards to the same maintained dependencies so
commands in the published report no longer install the vulnerable stack.

New runs use different software versions and are not exact reproductions of
the historical environment. The recorded results and their version metadata
remain unchanged.
