#!/bin/bash
# Evaluate reward-alignment (Pearson / Loss) for every checkpoint across
# ALL three data-quality groups in ONE pass per checkpoint:
#   successful_labeled  failure_labeled  suboptimal_labeled
#
# run_baseline_eval.py evaluates all quality groups automatically
# (custom_eval.py filter is disabled).  recompute_metrics.py then
# derives per-quality metric keys used by the combine_checkpoint_metric_plots.py.
#
# Output: baseline_eval_output/all_checkpoint_reward_alignment/

set -e

export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com

CHECKPOINT_BASE=/data/yingxi/robometer/logs/rbm
BASELINE_MODEL=/home/yingxi/RoboFAC/robometer/robometer/Robometer-4B
OUTPUT_BASE=/home/yingxi/RoboFAC/robometer/baseline_eval_output/all_checkpoint_reward_alignment
RECOMPUTE_SCRIPT=/home/yingxi/RoboFAC/robometer/evals/recompute_metrics.py
COMBINE_SCRIPT=/home/yingxi/RoboFAC/robometer/evals/combine_checkpoint_metric_plots.py
ROBOMETER_DIR=/home/yingxi/RoboFAC/robometer

DATASET=local_PegInsertionVertical_eval

COMMON_ARGS=(
    reward_model=rbm
    "custom_eval.eval_types=[reward_alignment]"
    "custom_eval.reward_alignment=[${DATASET}]"
    custom_eval.use_frame_steps=true
    custom_eval.subsample_n_frames=10
    custom_eval.reward_alignment_max_trajectories=30
    max_frames=8
    model_config.batch_size=16
)

mkdir -p "$OUTPUT_BASE"

# Check if a run dir already has results for all three quality groups
_all_qualities_done() {
    local results_file="$1/reward_alignment/${DATASET}_results.json"
    [ -f "$results_file" ] && python3 -c "
import json, sys
try:
    r = json.load(open('$results_file'))
    qs = set(x.get('quality_label') for x in r)
    sys.exit(0 if len(qs) >= 3 else 1)
except: sys.exit(1)
" 2>/dev/null
}

# ------------------------------------------------------------------
# 1. Robometer-4B baseline
# ------------------------------------------------------------------
BASELINE_OUT="$OUTPUT_BASE/baseline_Robometer-4B"
if _all_qualities_done "$BASELINE_OUT"; then
    echo "=== Skipping Robometer-4B baseline (all quality groups done) ==="
else
    echo ""
    echo "=== Evaluating Robometer-4B baseline (all 3 quality groups) ==="
    cd "$ROBOMETER_DIR"
    uv run python robometer/evals/run_baseline_eval.py \
        "${COMMON_ARGS[@]}" \
        model_path="$BASELINE_MODEL" \
        output_dir="$BASELINE_OUT"
    echo "Done: baseline → $BASELINE_OUT"
fi

# ------------------------------------------------------------------
# 2. Fine-tuned checkpoints
# ------------------------------------------------------------------
echo ""
echo "Checkpoints to evaluate:"
ls "$CHECKPOINT_BASE" | grep '^checkpoint-' | sort -V

for CKPT_DIR in "$CHECKPOINT_BASE"/checkpoint-*/; do
    CKPT_NAME=$(basename "$CKPT_DIR")
    CKPT_OUT="$OUTPUT_BASE/$CKPT_NAME"

    if _all_qualities_done "$CKPT_OUT"; then
        echo "=== Skipping $CKPT_NAME (all quality groups done) ==="
        continue
    fi

    echo ""
    echo "=== Evaluating $CKPT_NAME ==="
    cd "$ROBOMETER_DIR"
    uv run python robometer/evals/run_baseline_eval.py \
        "${COMMON_ARGS[@]}" \
        model_path="$CKPT_DIR" \
        output_dir="$CKPT_OUT"
    echo "Done: $CKPT_NAME"
done

# ------------------------------------------------------------------
# 3. Per-quality metric derivation
# ------------------------------------------------------------------
echo ""
echo "=== Recomputing per-quality metrics... ==="
cd "$ROBOMETER_DIR"
uv run python "$RECOMPUTE_SCRIPT" --output-base "$OUTPUT_BASE"

# ------------------------------------------------------------------
# 4. One metric figure per quality group
# ------------------------------------------------------------------
echo ""
echo "=== Creating per-quality metric plots... ==="
uv run python "$COMBINE_SCRIPT" \
    --output-base "$OUTPUT_BASE" \
    --baseline-dir "$OUTPUT_BASE/baseline_Robometer-4B"

echo ""
echo "=== Done! ==="
echo "  Output dir   : $OUTPUT_BASE"
echo "  Metric plots : $OUTPUT_BASE/metric_vs_step_*.png"
