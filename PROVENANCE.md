# Provenance Record — Hakuli

This file documents the development and provenance process of the Hakuli
optimizer, as referenced by the paper's Clean-Room Methodology section.

## Statement

Hakuli was implemented independently from the published APOLLO algorithmic
description (Zhu et al., *APOLLO: SGD-like Memory, AdamW-level Performance*,
MLSys 2025, arXiv:2412.05270) and its public documentation. No source code
from the official APOLLO implementation was used in the implementation of
Hakuli.

## Process record

| Phase | Date (2026) | Description |
|---|---|---|
| Specification review | 2026-10 | Public paper (arXiv:2412.05270) and public repository documentation read. No reference source code consulted. |
| Implementation | 2026-10 | Optimizer written from the mathematical specification. Validated numerically against an internal, specification-derived reference implementation. |
| Public release | 2026-10-09 | Initial public commit `a8ef673` of the Hakuli repository (Apache-2.0). |
| Black-box validation | 2026-10-09 | First execution of the official `apollo-torch` package (v1.0.3), used strictly as an external black box in the benchmark harness for the comparison reported in the paper. |

## Rules followed

1. The official implementation is never linked, copied, vendored, or
   redistributed by Hakuli. It is only imported by the benchmark harness at
   run time for black-box measurements.
2. All academic attributions (APOLLO, GaLore, Fira, Q-GaLore) are recorded in
   `NOTICE`, `CITATION.cff`, and the paper references.
3. No algorithmic-novelty claims are made for the update rule; Hakuli is
   presented as an independent implementation of the APOLLO method.

## License context

- Hakuli: Apache License 2.0.
- Official APOLLO repository: CC-BY-NC, with portions from GaLore under
  Apache-2.0. The Hugging Face Trainer integration of APOLLO requires the
  `apollo-torch` package (CC-BY-NC).
- GaLore and Fira: Apache-2.0.

The Apache-2.0 license of Hakuli grants no rights to third-party patents and
covers only the rights held by Hakuli's contributors over their own
contributions. A formal freedom-to-operate analysis is out of scope of this
record; no third-party patents related to the method were identified during
preparation.
