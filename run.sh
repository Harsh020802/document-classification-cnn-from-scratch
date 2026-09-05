#!/bin/bash
# Run any project script with the right environment, without remembering to activate.
#   ./run.sh scripts/export_samples.py --n 20
#   ./run.sh scripts/predict.py --test-index 0
PY=/opt/miniconda3/envs/docclf/bin/python
[ -x "$PY" ] || { echo "conda env 'docclf' not found at $PY"; exit 1; }
cd "$(dirname "$0")" || exit 1
PYTHONPATH=. exec "$PY" "$@"
