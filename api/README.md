# MythosForge API — Quick Start

## Installation

```bash
pip install -r api/requirements.txt
```

## Run

```bash
# Data-only mode (no PyTorch needed)
cd api && python -m app

# Or with uvicorn directly:
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## With OpenMythos Inference

```bash
# Install PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install OpenMythos
git clone https://github.com/kyegomez/OpenMythos.git
cd OpenMythos && pip install -r requirements.txt && pip install -e .

# Run the API with inference support
cd ../mythosforge-repo && uvicorn api.app:app --reload
```

## Docker

```bash
# Data-only mode (lightweight)
docker build -f api/Dockerfile -t mythosforge-api .
docker run -p 8000:8000 mythosforge-api

# Full mode with PyTorch + OpenMythos inference
docker build -f api/Dockerfile.full -t mythosforge-api-full .
docker run -p 8000:8000 mythosforge-api-full
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API root with endpoint links |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/info` | Project info |
| GET | `/api/v1/architecture` | Architecture details |
| GET | `/api/v1/components` | All components |
| GET | `/api/v1/components/{slug}` | Specific component |
| GET | `/api/v1/validation` | Validation results |
| GET | `/api/v1/roadmap` | Project roadmap |
| GET | `/api/v1/references` | Academic references |
| GET | `/api/v1/references/{id}` | Specific reference |
| GET | `/api/v1/i18n/{lang}` | Translations (es/en) |
| POST | `/api/v1/inference` | Model inference |
| GET | `/api/v1/inference/status` | Inference deps check |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc documentation |

## OpenAPI Docs

When the API is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json
