# Security Policy

## Supported versions

The latest released minor version receives security fixes.

| Version | Supported |
|---------|-----------|
| 2.x     | Yes       |
| 1.x     | No        |

## PyTorch dependency advisories

Hakuli 2.x requires a patched PyTorch installation. New installations must
use `torch>=2.15.0` with Python 3.11+.

| Advisory | Impact | Fixed in PyTorch |
| --- | --- | --- |
| [CVE-2025-32434](https://github.com/advisories/GHSA-53q9-r3pm-6pq6) | Code execution via malicious checkpoint | 2.6.0 |
| [CVE-2026-24747](https://github.com/advisories/GHSA-63cw-57p8-fm3p) | Code execution via malicious checkpoint | 2.10.0 |
| [GHSA-v32g-mr63-3j83](https://github.com/pytorch/pytorch/security/advisories/GHSA-v32g-mr63-3j83) | File disclosure via malicious checkpoint | 2.15.0 |

These advisories affect upstream PyTorch checkpoint loading, not Hakuli's
optimizer implementation. Do not load checkpoints from untrusted sources.

The benchmark reproduction environment is pinned to the dependencies used
for the October 2026 results and is not suitable for production or
untrusted inputs. Run historical reproductions in an isolated environment.

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
