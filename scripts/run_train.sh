#!/usr/bin/env bash
set -euo pipefail
STAMP=$(date +%Y%m%d_%H%M%S)
docker compose run --rm train \
  python -m yourpkg.cli --config configs/default.yaml --out experiments/runs/${STAMP}
