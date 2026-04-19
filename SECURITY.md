# 🔒 Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | ✅        |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in this project, please follow these steps:

1. **Do NOT** open a public issue — this helps prevent malicious actors from exploiting the vulnerability before a fix is available.
2. Instead, send a private report via [GitHub Security Advisories](https://github.com/smouj/MythosForge/security/advisories/new).
3. Include as much detail as possible: steps to reproduce, affected component, potential impact, and any suggested fix.
4. We will acknowledge receipt within **48 hours** and aim to provide a resolution or timeline within **7 days**.

## What Qualifies as a Vulnerability?

- Code execution vulnerabilities in source files (`src/`, `docs/`)
- Supply chain attacks via dependency injection
- Cross-site scripting (XSS) in the GitHub Pages site (`docs/`)
- Sensitive data exposure (secrets, tokens, keys)
- Path traversal or file inclusion issues

## What Does NOT Qualify

- Theoretical concerns about the transformer architecture itself
- Performance degradation issues
- Documentation inaccuracies (open a regular issue instead)

## Responsible Disclosure

We follow [Coordinated Disclosure](https://github.com/securitylab) principles. After a fix is released, credit will be given to the reporter in the relevant commit or release notes (unless anonymity is requested).

## Dependency Security

This project uses minimal dependencies to reduce attack surface:
- The GitHub Pages site (`docs/`) is pure HTML/CSS/JS with no build step
- Python scripts in `src/` use only PyTorch and standard library
- No API keys, secrets, or tokens are committed to the repository
