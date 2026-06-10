#!/bin/bash
# Run run_baseline_eval.py (with ground-truth metrics) for every checkpoint
# under /data/yingxi/robometer/logs/rbm/ AND for the Robometer-4B baseline,
# then produce combined per-trajectory + metric-vs-step plots.

set -e

export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com

CHECKPOINT_BASE=/data/yingxi/robometer/logs/rbm
BASELINE_MODEL=/home/yingxi/RoboFAC/robometer/robometer/Robometer-4B
OUTPUT_BASE=/data/yingxi/robometer/all_checkpoint_eval_output_metrics
COMBINE_SCRIPT=/home/yingxi/RoboFAC/robometer/evals/combine_checkpoint_metric_plots.py
ROBOMETER_DIR=/home/yingxi/RoboFAC/robometer

# Shared eval params (mirror the original command)
COMMON_ARGS=(
    reward_model=rbm
    "custom_eval.eval_types=[reward_alignment]"
    "custom_eval.reward_alignment=[local_eval_real_success]"
    custom_eval.use_frame_steps=true
    custom_eval.subsample_n_frames=10
    custom_eval.reward_alignment_max_trajectories=30
    max_frames=8
    model_config.batch_size=16
)

mkdir -p "$OUTPUT_BASE"

# ------------------------------------------------------------------
# 1. Robometer-4B baseline
# ------------------------------------------------------------------
BASELINE_OUT="$OUTPUT_BASE/baseline_Robometer-4B"
if [ -d "$BASELINE_OUT/reward_alignment" ]; then
    echo "=== Skipping Robometer-4B baseline (already evaluated) ==="
else
    echo ""
    echo "=== Evaluating Robometer-4B baseline ==="
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

    if [ -d "$CKPT_OUT/reward_alignment" ]; then
        echo "=== Skipping $CKPT_NAME (already evaluated) ==="
        continue
    fi

    echo ""
    echo "=== Evaluating $CKPT_NAME ==="
    echo "    checkpoint : $CKPT_DIR"
    echo "    output dir : $CKPT_OUT"

    cd "$ROBOMETER_DIR"
    uv run python robometer/evals/run_baseline_eval.py \
        "${COMMON_ARGS[@]}" \
        model_path="$CKPT_DIR" \
        output_dir="$CKPT_OUT"

    echo "Done: $CKPT_NAME"
done

# ------------------------------------------------------------------
# 3. Combined plots
# ------------------------------------------------------------------
echo ""
echo "=== All evals complete. Creating combined plots... ==="
cd "$ROBOMETER_DIR"
uv run python "$COMBINE_SCRIPT" \
    --output-base "$OUTPUT_BASE" \
    --baseline-dir "$BASELINE_OUT"

echo ""
echo "=== Done! ==="
echo "  Per-trajectory plots : $OUTPUT_BASE/combined_trajectory_plots/"
echo "  Metric summary plot  : $OUTPUT_BASE/metric_vs_training_step.png"
