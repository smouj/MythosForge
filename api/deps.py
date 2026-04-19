"""
MythosForge API — FastAPI dependencies.

Reusable dependency functions for health checks, auth, etc.
"""

from __future__ import annotations

import os
import sys
import time
import logging

from api import __version__, __api_version__
from api.models import HealthResponse
from api.settings import settings

logger = logging.getLogger(__name__)

# ─── Ensure mythosforge is importable ────────────────

_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def get_health_data() -> dict:
    """Gather dependency availability for health endpoint."""
    pytorch_available = False
    mythosforge_available = False
    pytorch_version = None

    try:
        import torch
        pytorch_available = True
        pytorch_version = torch.__version__
    except ImportError:
        pass

    try:
        from mythosforge import OpenMythos  # noqa: F401
        mythosforge_available = True
    except ImportError:
        pass

    return {
        "pytorch_available": pytorch_available,
        "pytorch_version": pytorch_version,
        "mythosforge_available": mythosforge_available,
    }


def build_health_response(uptime_seconds: float) -> HealthResponse:
    """Build the HealthResponse with current dependency status."""
    data = get_health_data()
    return HealthResponse(
        status="healthy",
        version=__version__,
        api_version=__api_version__,
        pytorch_available=data["pytorch_available"],
        openmythos_available=data["mythosforge_available"],
        uptime_seconds=round(uptime_seconds, 2),
    )
