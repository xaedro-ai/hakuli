# Security Policy

## Supported versions

The latest released minor version receives security fixes.

| Version | Supported |
|---------|-----------|
| 2.x     | Yes       |
| 1.x     | Yes       |

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
