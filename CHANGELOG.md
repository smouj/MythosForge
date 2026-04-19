# Changelog — MythosForge

All notable changes to the MythosForge project are documented in this file.

## [0.5.0] — 2026-04-19

### Security
- **Fixed**: Inference error handler no longer exposes `str(e)` to clients — returns generic `"inference_failed"` instead
- **Added**: Safe error handlers (`api/errors.py`) — unhandled exceptions return `error_id` + `request_id` without stack traces or internal details
- **Added**: API key authentication module (`api/security/auth.py`) — supports Bearer token and X-API-Key header, disabled by default

### Architecture
- **Refactored**: Catalog routes extracted from `app.py` into `api/routers/public.py` for cleaner separation of concerns
- **Added**: `api/settings.py` — all configuration centralized via Pydantic Settings with `MYTHOSFORGE_` env prefix
- **Added**: `api/deps.py` — reusable FastAPI dependencies (health check builder)
- **Added**: `api/observability/metrics.py` — in-process Prometheus-style metrics (counters, histograms, gauges)
- **App.py** now serves as a thin orchestration layer: CORS, error handlers, middleware, and system routes only

### Configuration
- **Changed**: CORS origins restricted to allowlist (no more `["*"]` with credentials)
- **Added**: `.env.example` with documented configuration variables
- **Added**: Request ID middleware (`X-Request-ID` header on all responses)
- **Added**: Structured logging with configurable level and format

### Testing
- **Added**: `api/tests/test_api.py` — 39 integration tests covering all endpoints, security, and error handling
- **Added**: CI pipeline runs API tests on every push/PR
- **Tests cover**: root, health, info, architecture, components, validation, roadmap, references, i18n, metrics, security, inference validation, OpenAPI contract

### CI/CD
- **Added**: Ruff lint job (check + format) in CI pipeline
- **Added**: OpenAPI snapshot saved as CI artifact on every run
- **Improved**: CI now runs lint and tests in parallel, both must pass before build

### Documentation
- **Added**: `CHANGELOG.md`
- **Updated**: `README.md` reflects new architecture, security features, and testing

## [0.4.0] — 2026-04-19

### Added
- API key authentication (optional, disabled by default)
- In-process metrics endpoint (`/api/v1/metrics`)
- Safe error handlers without internal detail leaks
- Request ID tracking middleware
- 39 API integration tests
- Pydantic Settings for environment-based configuration
- CORS restricted to allowlist

## [0.3.0] — 2026-04-19

### Added
- Self-contained `mythosforge` PyTorch package (`src/mythosforge/`)
- Real inference endpoint using built-in mythosforge package
- GQA and MLA attention switchable via configuration
- 30 PyTorch model tests (all passing)
- Model status endpoint (`/api/v1/inference/status`)

## [0.2.0] — 2026-04-19

### Added
- REST API with FastAPI and OpenAPI documentation
- Project info, architecture, components, validation, roadmap, references endpoints
- i18n translations (ES/EN)
- Pydantic v2 schemas for all endpoints

## [0.1.0] — 2026-04-19

### Added
- GitHub repository with README, badges, and ASCII architecture diagram
- GitHub Pages site with dark theme and i18n (ES/EN)
- Project logo, favicon, and social banner
- Screenshots for all documentation sections
- CI/CD pipeline for Pages deployment
- CONTRIBUTING.md, SECURITY.md, LICENSE (MIT)
- Issue templates, PR template, CODEOWNERS, FUNDING.yml, Dependabot
