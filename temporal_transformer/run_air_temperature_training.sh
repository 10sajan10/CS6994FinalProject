#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run temporal_transformer/setup_env.sh first."
  exit 1
fi

source .venv/bin/activate
python temporal_transformer/train_air_temperature_transformer.py "$@"
