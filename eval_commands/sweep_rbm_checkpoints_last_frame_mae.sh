#!/bin/bash
# Evaluate last-frame MAE for every checkpoint across all quality groups:
#   successful_labeled, failure_labeled, suboptimal_labeled
#
# Each quality group produces metrics with a quality-prefix key, e.g.
#   successful_labeled/last_frame_mae_full
# All runs append into the same <output_dir>/last_frame_mae/metrics.json
# so the combine script can plot separate lines per quality.

set -e

export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com

CHECKPOINT_BASE=/data/yingxi/robometer/logs/rbm
BASELINE_MODEL=/home/yingxi/RoboFAC/robometer/robometer/Robometer-4B
OUTPUT_BASE=/home/yingxi/RoboFAC/robometer/baseline_eval_output/all_checkpoint_last_frame_mae
EVAL_SCRIPT=/home/yingxi/RoboFAC/robometer/evals/run_last_frame_mae_eval.py
COMBINE_SCRIPT=/home/yingxi/RoboFAC/robometer/evals/combine_checkpoint_metric_plots.py
ROBOMETER_DIR=/home/yingxi/RoboFAC/robometer

DATASET=local_PegInsertionVertical_eval
N_SEEDS=5
BATCH_SIZE=16
RANDOM_SEED=42
MAX_FRAMES=8

# Quality groups to evaluate
QUALITY_GROUPS=(successful_labeled failure_labeled suboptimal_labeled)

mkdir -p "$OUTPUT_BASE"

# Helper: check if a quality-prefixed key already exists in metrics.json
_quality_done() {
    local metrics_file="$1/last_frame_mae/metrics.json"
    local quality="$2"
    [ -f "$metrics_file" ] && \
        python3 -c "
import json, sys
try:
    d = json.load(open('$metrics_file'))
    key = '${quality}/last_frame_mae_full'
    sys.exit(0 if key in d else 1)
except: sys.exit(1)
" 2>/dev/null
}

# Evaluate one (model, quality_group) combination
_run_eval() {
    local model_path="$1"
    local out_dir="$2"
    local quality="$3"

    if _quality_done "$out_dir" "$quality"; then
        echo "  [skip] $(basename $out_dir) / $quality (already done)"
        return
    fi
    echo "  [eval] $(basename $out_dir) / $quality"
    cd "$ROBOMETER_DIR"
    uv run python "$EVAL_SCRIPT" \
        --model-path "$model_path" \
        --output-dir "$out_dir" \
        --dataset-name "$DATASET" \
        --quality-labels "$quality" \
        --n-seeds "$N_SEEDS" \
        --batch-size "$BATCH_SIZE" \
        --random-seed "$RANDOM_SEED" \
        --max-frames "$MAX_FRAMES"
}

# ------------------------------------------------------------------
# Main loop: for each quality group × (baseline + checkpoints)
# ------------------------------------------------------------------
for QUALITY in "${QUALITY_GROUPS[@]}"; do
    echo ""
    echo "====== Quality group: $QUALITY ======"

    # Baseline
    _run_eval "$BASELINE_MODEL" "$OUTPUT_BASE/baseline_Robometer-4B" "$QUALITY"

    # Fine-tuned checkpoints
    for CKPT_DIR in "$CHECKPOINT_BASE"/checkpoint-*/; do
        CKPT_NAME=$(basename "$CKPT_DIR")
        _run_eval "$CKPT_DIR" "$OUTPUT_BASE/$CKPT_NAME" "$QUALITY"
    done
done

# ------------------------------------------------------------------
# Combined plot
# ------------------------------------------------------------------
echo ""
echo "=== All evals complete. Creating combined plots... ==="
cd "$ROBOMETER_DIR"
uv run python "$COMBINE_SCRIPT" \
    --output-base "$OUTPUT_BASE" \
    --baseline-dir "$OUTPUT_BASE/baseline_Robometer-4B"

echo ""
echo "=== Done! ==="
echo "  Output dir   : $OUTPUT_BASE"
echo "  Metric plots : $OUTPUT_BASE/metric_vs_step_*.png"
