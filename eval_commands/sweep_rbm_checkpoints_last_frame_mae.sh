#!/bin/bash
# Evaluate last-frame MAE for every checkpoint across all quality groups:
#   successful_labeled, failure_labeled, suboptimal_labeled
#
# Usage:
#   # sequential
#   bash eval_commands/sweep_rbm_checkpoints_last_frame_mae.sh
#
#   # parallel across 3 GPUs (models within each quality group run concurrently)
#   bash eval_commands/sweep_rbm_checkpoints_last_frame_mae.sh --gpus 5,6,7

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
QUALITY_GROUPS=(successful_labeled failure_labeled suboptimal_labeled)

# ---- CLI parsing ------------------------------------------------------------
GPU_IDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) IFS=',' read -ra GPU_IDS <<< "$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--gpus <gpu-ids>]  e.g. --gpus 5,6,7"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_BASE"
LOG_DIR="$OUTPUT_BASE/eval_logs"
mkdir -p "$LOG_DIR"

# ---- GPU dispatcher (only active with --gpus) -------------------------------
if [ "${#GPU_IDS[@]}" -gt 0 ]; then
    echo "Multi-GPU mode: GPUs = ${GPU_IDS[*]}  (${#GPU_IDS[@]} slots)"

    declare -A _SLOT_PIDS
    for _g in "${GPU_IDS[@]}"; do _SLOT_PIDS[$_g]=0; done

    _acquire_gpu_slot() {
        while true; do
            for _g in "${GPU_IDS[@]}"; do
                local _pid="${_SLOT_PIDS[$_g]}"
                if [[ "$_pid" -eq 0 ]] || ! kill -0 "$_pid" 2>/dev/null; then
                    _SLOT_PIDS[$_g]=0
                    echo "$_g"
                    return 0
                fi
            done
            sleep 2
        done
    }

    _wait_all_slots() {
        wait
        for _g in "${GPU_IDS[@]}"; do _SLOT_PIDS[$_g]=0; done
    }

    _run_eval() {
        local model_path="$1" out_dir="$2" quality="$3"
        if _quality_done "$out_dir" "$quality"; then
            echo "  [skip] $(basename "$out_dir") / $quality"
            return 0
        fi
        local gpu
        gpu=$(_acquire_gpu_slot)
        local log_file="$LOG_DIR/$(basename "$out_dir")__${quality}.log"
        echo "  [GPU ${gpu}] start: $(basename "$out_dir") / ${quality}"
        (
            cd "$ROBOMETER_DIR"
            CUDA_VISIBLE_DEVICES="$gpu" uv run python "$EVAL_SCRIPT" \
                --model-path     "$model_path" \
                --output-dir     "$out_dir" \
                --dataset-name   "$DATASET" \
                --quality-labels "$quality" \
                --n-seeds        "$N_SEEDS" \
                --batch-size     "$BATCH_SIZE" \
                --random-seed    "$RANDOM_SEED" \
                --max-frames     "$MAX_FRAMES"
            rc=$?
            if [ $rc -eq 0 ]; then
                echo "  [GPU ${gpu}] done: $(basename "$out_dir") / ${quality}"
            else
                echo "  [GPU ${gpu}] FAILED: $(basename "$out_dir") / ${quality} (exit ${rc})" >&2
            fi
            exit $rc
        ) >"$log_file" 2>&1 &
        _SLOT_PIDS[$gpu]=$!
    }

else
    echo "Single-GPU mode  (use --gpus to parallelise)"
    _wait_all_slots() { :; }

    _run_eval() {
        local model_path="$1" out_dir="$2" quality="$3"
        if _quality_done "$out_dir" "$quality"; then
            echo "  [skip] $(basename "$out_dir") / $quality"
            return 0
        fi
        echo "  [eval] $(basename "$out_dir") / $quality"
        cd "$ROBOMETER_DIR"
        uv run python "$EVAL_SCRIPT" \
            --model-path     "$model_path" \
            --output-dir     "$out_dir" \
            --dataset-name   "$DATASET" \
            --quality-labels "$quality" \
            --n-seeds        "$N_SEEDS" \
            --batch-size     "$BATCH_SIZE" \
            --random-seed    "$RANDOM_SEED" \
            --max-frames     "$MAX_FRAMES"
    }
fi

# ---- completion check -------------------------------------------------------
_quality_done() {
    local metrics_file="$1/last_frame_mae/metrics.json"
    local quality="$2"
    [ -f "$metrics_file" ] && python3 -c "
import json, sys
try:
    d = json.load(open('$metrics_file'))
    key = '${quality}/last_frame_mae_full'
    sys.exit(0 if key in d else 1)
except: sys.exit(1)
" 2>/dev/null
}

# ---- main loop: quality groups serial; models parallel within each quality --
for QUALITY in "${QUALITY_GROUPS[@]}"; do
    echo ""
    echo "====== Quality group: $QUALITY ======"

    _run_eval "$BASELINE_MODEL" "$OUTPUT_BASE/baseline_Robometer-4B" "$QUALITY"

    for CKPT_DIR in "$CHECKPOINT_BASE"/checkpoint-*/; do
        CKPT_NAME=$(basename "$CKPT_DIR")
        _run_eval "$CKPT_DIR" "$OUTPUT_BASE/$CKPT_NAME" "$QUALITY"
    done

    echo "  Waiting for $QUALITY evals..."
    _wait_all_slots
    echo "  $QUALITY complete."
done

# ---- combine plots ----------------------------------------------------------
echo ""
echo "=== Creating combined plots... ==="
cd "$ROBOMETER_DIR"
uv run python "$COMBINE_SCRIPT" \
    --output-base "$OUTPUT_BASE" \
    --baseline-dir "$OUTPUT_BASE/baseline_Robometer-4B"

echo ""
echo "=== Done! ==="
echo "  Output : $OUTPUT_BASE"
echo "  Logs   : $LOG_DIR/"
echo "  Plots  : $OUTPUT_BASE/metric_vs_step_*.png"
