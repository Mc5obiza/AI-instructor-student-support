#!/bin/bash
set -euo pipefail

echo "Running tests..."

cd "$(dirname "$0")"
python -m pytest -v
