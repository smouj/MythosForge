#!/usr/bin/env bash
# Start the MythosForge API server
# Usage: bash run_api.sh

cd "$(dirname "$0")"
export PYTHONPATH=.

echo "=========================================="
echo "  MythosForge API v0.3.0"
echo "  Backend: mythosforge (built-in PyTorch)"
echo "=========================================="
echo ""
echo "Installing dependencies..."
pip install -q -r api/requirements.txt
echo ""
echo "Starting server on http://localhost:8000"
echo "Swagger UI: http://localhost:8000/docs"
echo ""
python -m api
