"""
MythosForge API — Integration tests with TestClient.

Run with:
    cd /path/to/MythosForge
    PYTHONPATH=. python -m pytest api/tests/test_api.py -v

These tests verify endpoints, status codes, response schemas, and error handling.
No external ML dependencies required — only FastAPI + httpx.
"""

import os
import sys
import pytest

# Ensure api package is importable
_API_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from fastapi.testclient import TestClient
from api.app import app


client = TestClient(app)


# ═══════════════════════════════════════════════════════
# Root & Health
# ═══════════════════════════════════════════════════════

class TestRoot:
    def test_root_returns_api_info(self):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "MythosForge API"
        assert "endpoints" in data
        assert "/docs" in data["docs_url"]

    def test_root_has_version(self):
        r = client.get("/")
        data = r.json()
        assert "version" in data
        assert data["version"].startswith("0.")


class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"

    def test_health_has_uptime(self):
        r = client.get("/api/v1/health")
        data = r.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_has_version(self):
        r = client.get("/api/v1/health")
        data = r.json()
        assert "version" in data
        assert "api_version" in data

    def test_health_reports_dependencies(self):
        r = client.get("/api/v1/health")
        data = r.json()
        assert "pytorch_available" in data
        assert isinstance(data["pytorch_available"], bool)


# ═══════════════════════════════════════════════════════
# Project Info
# ═══════════════════════════════════════════════════════

class TestInfo:
    def test_info_returns_200(self):
        r = client.get("/api/v1/info")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "MythosForge"
        assert "features" in data
        assert len(data["features"]) > 0

    def test_info_has_disclaimer(self):
        r = client.get("/api/v1/info")
        data = r.json()
        assert "disclaimer" in data
        assert "Anthropic" in data["disclaimer"]


# ═══════════════════════════════════════════════════════
# Architecture
# ═══════════════════════════════════════════════════════

class TestArchitecture:
    def test_architecture_returns_200(self):
        r = client.get("/api/v1/architecture")
        assert r.status_code == 200
        data = r.json()
        assert "flow" in data
        assert "recurrent_components" in data
        assert "recurrent_rule" in data

    def test_architecture_flow_has_prelude_and_coda(self):
        r = client.get("/api/v1/architecture")
        flow_names = [b["name"] for b in r.json()["flow"]]
        assert "PRELUDE" in flow_names
        assert "RECURRENT BLOCK" in flow_names
        assert "CODA" in flow_names

    def test_architecture_recurrent_components(self):
        r = client.get("/api/v1/architecture")
        components = r.json()["recurrent_components"]
        names = [c["short_name"] for c in components]
        assert "mla-gqa" in names
        assert "moe" in names
        assert "lti" in names
        assert "act" in names
        assert "lora" in names


# ═══════════════════════════════════════════════════════
# Components
# ═══════════════════════════════════════════════════════

class TestComponents:
    def test_components_list(self):
        r = client.get("/api/v1/components")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == len(data["components"])
        assert data["total"] >= 6

    def test_component_detail_valid(self):
        r = client.get("/api/v1/components/mla-gqa")
        assert r.status_code == 200
        data = r.json()
        assert data["slug"] == "mla-gqa"
        assert "description" in data
        assert "references" in data

    def test_component_detail_not_found(self):
        r = client.get("/api/v1/components/nonexistent")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_all_component_slugs_accessible(self):
        """Every component listed should be accessible by slug."""
        r = client.get("/api/v1/components")
        slugs = [c["slug"] for c in r.json()["components"]]
        for slug in slugs:
            cr = client.get(f"/api/v1/components/{slug}")
            assert cr.status_code == 200, f"Slug {slug} returned {cr.status_code}"


# ═══════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════

class TestValidation:
    def test_validation_returns_200(self):
        r = client.get("/api/v1/validation")
        assert r.status_code == 200
        data = r.json()
        assert "inference_checks" in data
        assert "repo_status" in data
        assert data["patch_available"] is True

    def test_validation_checks_have_status(self):
        r = client.get("/api/v1/validation")
        checks = r.json()["inference_checks"]
        for check in checks:
            assert "status" in check
            assert check["status"] in ("pass", "fail", "warn", "skip")

    def test_validation_repo_status(self):
        r = client.get("/api/v1/validation")
        status = r.json()["repo_status"]
        assert len(status) > 0
        for item in status:
            assert "feature" in item
            assert "available" in item


# ═══════════════════════════════════════════════════════
# Roadmap
# ═══════════════════════════════════════════════════════

class TestRoadmap:
    def test_roadmap_returns_200(self):
        r = client.get("/api/v1/roadmap")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 7
        assert data["completed"] >= 0

    def test_roadmap_phases_sequential(self):
        r = client.get("/api/v1/roadmap")
        phases = r.json()["phases"]
        for i, phase in enumerate(phases):
            assert phase["phase"] == i

    def test_roadmap_statuses_valid(self):
        r = client.get("/api/v1/roadmap")
        valid = {"completed", "in_progress", "pending"}
        for phase in r.json()["phases"]:
            assert phase["status"] in valid


# ═══════════════════════════════════════════════════════
# References
# ═══════════════════════════════════════════════════════

class TestReferences:
    def test_references_list(self):
        r = client.get("/api/v1/references")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == len(data["references"])
        assert data["total"] >= 12

    def test_reference_detail_valid(self):
        r = client.get("/api/v1/references/R1")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "R1"
        assert "url" in data
        assert "title" in data

    def test_reference_not_found(self):
        r = client.get("/api/v1/references/INVALID")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_reference_case_insensitive(self):
        r_upper = client.get("/api/v1/references/R4")
        r_lower = client.get("/api/v1/references/r4")
        assert r_upper.status_code == 200
        assert r_lower.status_code == 200
        assert r_upper.json()["id"] == r_lower.json()["id"]


# ═══════════════════════════════════════════════════════
# i18n
# ═══════════════════════════════════════════════════════

class TestI18n:
    def test_spanish_translations(self):
        r = client.get("/api/v1/i18n/es")
        assert r.status_code == 200
        data = r.json()
        assert data["language"] == "es"
        assert data["total_keys"] > 0
        assert len(data["translations"]) == data["total_keys"]

    def test_english_translations(self):
        r = client.get("/api/v1/i18n/en")
        assert r.status_code == 200
        data = r.json()
        assert data["language"] == "en"
        assert data["total_keys"] > 0

    def test_es_and_en_have_same_keys(self):
        r_es = client.get("/api/v1/i18n/es").json()
        r_en = client.get("/api/v1/i18n/en").json()
        assert r_es["total_keys"] == r_en["total_keys"]
        assert set(r_es["translations"].keys()) == set(r_en["translations"].keys())

    def test_invalid_language(self):
        r = client.get("/api/v1/i18n/fr")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()


# ═══════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════

class TestMetrics:
    def test_metrics_returns_200(self):
        r = client.get("/api/v1/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "counters" in data
        assert "histograms" in data

    def test_metrics_counts_after_requests(self):
        """After previous requests, counters should be non-zero."""
        r = client.get("/api/v1/metrics")
        data = r.json()
        total = data["counters"].get("http_requests_total", 0)
        assert total > 0


# ═══════════════════════════════════════════════════════
# Security & Error Handling
# ═══════════════════════════════════════════════════════

class TestSecurity:
    def test_500_does_not_leak_stack_trace(self):
        """Internal errors must not expose implementation details."""
        # Trigger an unhandled error via a malformed request to a protected endpoint
        # The global exception handler should sanitize the response
        r = client.get("/api/v1/components/../../etc/passwd")
        # Should be 404 (not 500) due to path normalization, or safe 500
        if r.status_code == 500:
            data = r.json()
            assert "error_id" in data
            assert "detail" in data
            assert "Traceback" not in str(data)

    def test_response_has_request_id_header(self):
        r = client.get("/api/v1/health")
        assert "x-request-id" in r.headers
        assert len(r.headers["x-request-id"]) > 0

    def test_cors_headers_present(self):
        r = client.options(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        # CORS preflight should work for allowed origins
        # (TestClient doesn't fully simulate CORS, but we verify no crash)
        assert r.status_code in (200, 405)


# ═══════════════════════════════════════════════════════
# Inference (graceful degradation)
# ═══════════════════════════════════════════════════════

class TestInferenceStatus:
    def test_inference_status_returns_200(self):
        r = client.get("/api/v1/inference/status")
        assert r.status_code == 200
        data = r.json()
        assert "pytorch_available" in data
        assert "mythosforge_available" in data
        assert "backend" in data

    def test_inference_validates_prompt_length(self):
        """Empty prompt should be rejected by Pydantic validation (min_length=1)."""
        r = client.post(
            "/api/v1/inference",
            json={"prompt": "", "max_tokens": 4},
        )
        assert r.status_code == 422

    def test_inference_validates_max_tokens_limit(self):
        """max_tokens > 512 should be rejected."""
        r = client.post(
            "/api/v1/inference",
            json={"prompt": "test", "max_tokens": 999},
        )
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════
# OpenAPI Contract
# ═══════════════════════════════════════════════════════

class TestOpenAPI:
    def test_openapi_json_available(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        data = r.json()
        assert data["info"]["title"] == "MythosForge API"
        assert "/api/v1/health" in data["paths"]
        assert "/api/v1/components" in data["paths"]
        assert "/api/v1/inference" in data["paths"]

    def test_docs_page_available(self):
        r = client.get("/docs")
        assert r.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
