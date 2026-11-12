# Security Policy

## Supported versions

The latest released minor version receives security fixes.

| Version | Supported |
|---------|-----------|
| 2.x     | Yes       |
| 1.x     | No        |

## Dependency security baseline

Current supported Hakuli installs require Python 3.11+ and
`torch>=2.15.0`. Earlier torch releases include checkpoint deserialization
vulnerabilities, including:

- CVE-2025-32434 / GHSA-53q9-r3pm-6pq6 (fixed in torch 2.6.0)
- CVE-2026-24747 / GHSA-63cw-57p8-fm3p (fixed in torch 2.10.0)
- GHSA-v32g-mr63-3j83 (torch 2.14.x arbitrary file read; fixed in 2.15.0)

These are **upstream PyTorch vulnerabilities**, not reports of a flaw in
Hakuli's optimizer algorithm. Do not load untrusted checkpoint files or model
artifacts even when using `torch.load(..., weights_only=True)`.

`benchmarks/requirements-repro.txt` and the reproduction Dockerfile preserve
historical dependency versions to make the October 2026 measurements
reproducible; they are intentionally **not** supported for production and
include vulnerable packages. Use only in an isolated, disposable environment
without credentials, mounted secrets, or untrusted model files.

## Reporting a vulnerability

Please do not report security vulnerabilities through public issues.

Use GitHub's private vulnerability reporting on this repository
("Security" tab → "Report a vulnerability"). Include:

- a description of the issue and its impact,
- a minimal reproduction (code snippet, configuration),
- the affected version(s) and environment.

You can expect an acknowledgement within a few business days. We will keep
you informed of the remediation progress and credit you in the release notes
unless you prefer otherwise.

## Scope

Hakuli is a training-time optimizer library. Relevant security concerns
include, for example, unsafe deserialization paths, code execution during
checkpoint loading, or dependency vulnerabilities. Model quality and
numerical behaviour are tracked as regular bugs, not security issues.
