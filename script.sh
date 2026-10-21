#!/bin/bash
set -euo pipefail

echo "Running tests..."

cd backend/test
python3 -m pytest -v
