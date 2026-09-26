#!/usr/bin/env bash
set -euo pipefail
SETTING=${SETTING:-both}
IND_ROOT=${IND_ROOT:?set IND_ROOT to the latest nine-task test HF dataset/root}
OOD_ROOT=${OOD_ROOT:?set OOD_ROOT to the OOD test HF dataset/root}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/yingxi/robometer/setting_ab_eval}
PYTHON_BIN=${PYTHON_BIN:-/data/yingxi/RoboFPE/robometer/.venv/bin/python}
export PYTHONPATH=/data/yingxi/RoboFPE/robometer:/data/yingxi/RoboFPE${PYTHONPATH:+:$PYTHONPATH}
exec "$PYTHON_BIN" /data/yingxi/RoboFPE/robometer/evals/eval_setting_ab_ind_ood.py \
  --setting "$SETTING" --ind-root "$IND_ROOT" --ood-root "$OOD_ROOT" \
  --output-root "$OUTPUT_ROOT" --top-k "${TOP_K:-50}" \
  --batch-size "${BATCH_SIZE:-16}" --max-frames "${MAX_FRAMES:-8}" "$@"
