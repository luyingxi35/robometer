#!/bin/bash
# Run run_pred_visualization_no_labels.py for every checkpoint under
# /data/yingxi/robometer/logs/rbm/ and then combine all results into
# one plot per trajectory.

set -e

export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com

CHECKPOINT_BASE=/data/yingxi/robometer/logs/rbm
OUTPUT_BASE=/data/yingxi/robometer/all_checkpoint_eval_output
COMBINE_SCRIPT=/home/yingxi/RoboFAC/robometer/evals/combine_checkpoint_eval_plots.py
ROBOMETER_DIR=/home/yingxi/RoboFAC/robometer

mkdir -p "$OUTPUT_BASE"

echo "Checkpoints to evaluate:"
ls "$CHECKPOINT_BASE" | grep '^checkpoint-' | sort -V

for CKPT_DIR in "$CHECKPOINT_BASE"/checkpoint-*/; do
    CKPT_NAME=$(basename "$CKPT_DIR")
    CKPT_OUT="$OUTPUT_BASE/$CKPT_NAME"

    # Skip if results already exist (allows resuming an interrupted run)
    if [ -d "$CKPT_OUT/oracle" ]; then
        echo "=== Skipping $CKPT_NAME (already evaluated) ==="
        continue
    fi

    echo ""
    echo "=== Evaluating $CKPT_NAME ==="
    echo "    checkpoint : $CKPT_DIR"
    echo "    output dir : $CKPT_OUT"

    cd "$ROBOMETER_DIR"
    uv run python robometer/evals/run_pred_visualization_no_labels.py \
        reward_model=rbm \
        model_path="$CKPT_DIR" \
        custom_eval.use_frame_steps=true \
        custom_eval.subsample_n_frames=10 \
        custom_eval.reward_alignment_max_trajectories=30 \
        max_frames=8 \
        model_config.batch_size=16 \
        output_dir="$CKPT_OUT"

    echo "Done: $CKPT_NAME"
done

echo ""
echo "=== All checkpoints evaluated. Creating combined plots... ==="
cd "$ROBOMETER_DIR"
uv run python "$COMBINE_SCRIPT" --output-base "$OUTPUT_BASE"
echo ""
echo "=== Done! Combined plots saved to: $OUTPUT_BASE/combined_trajectory_plots/ ==="
