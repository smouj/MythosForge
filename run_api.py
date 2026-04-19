#!/usr/bin/env python3
"""
run_api.sh — Start the MythosForge API server.

Usage:
    ./run_api.sh
    # or:
    bash run_api.sh
    # or:
    python run_api.py
"""

# run_api.sh content:
# cd "$(dirname "$0")"
# pip install -r api/requirements.txt -q
# PYTHONPATH=. python -m api

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("  MythosForge API v0.3.0")
print("  Backend: mythosforge v0.1.0 (built-in PyTorch)")
print("=" * 50)
print()

# Check if running from this script
if __name__ == "__main__":
    os.environ["PYTHONPATH"] = "."
    os.execvp(sys.executable, [sys.executable, "-m", "api"])
