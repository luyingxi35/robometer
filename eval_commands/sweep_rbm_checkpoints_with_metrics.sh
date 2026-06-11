#!/bin/bash
# Evaluate reward-alignment (Pearson / Loss) for every checkpoint across
# ALL three data-quality groups in ONE pass per checkpoint.
#
# Usage:
#   # sequential
#   bash eval_commands/sweep_rbm_checkpoints_with_metrics.sh
#
#   # parallel across 3 GPUs
#   bash eval_commands/sweep_rbm_checkpoints_with_metrics.sh --gpus 5,6,7

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

    _run_model_eval() {
        local model_path="$1" out_dir="$2" label="$3"
        local gpu
        gpu=$(_acquire_gpu_slot)
        local log_file="$LOG_DIR/${label}.log"
        echo "  [GPU ${gpu}] start: ${label}  log: $(basename "${log_file}")"
        (
            cd "$ROBOMETER_DIR"
            CUDA_VISIBLE_DEVICES="$gpu" uv run python robometer/evals/run_baseline_eval.py \
                "${COMMON_ARGS[@]}" \
                model_path="$model_path" \
                output_dir="$out_dir"
            rc=$?
            if [ $rc -eq 0 ]; then
                echo "  [GPU ${gpu}] done: ${label}"
            else
                echo "  [GPU ${gpu}] FAILED: ${label} (exit ${rc})" >&2
            fi
            exit $rc
        ) >"$log_file" 2>&1 &
        _SLOT_PIDS[$gpu]=$!
    }

else
    echo "Single-GPU mode  (use --gpus to parallelise)"
    _wait_all_slots() { :; }

    _run_model_eval() {
        local model_path="$1" out_dir="$2" label="$3"
        echo ""
        echo "=== Evaluating $label ==="
        cd "$ROBOMETER_DIR"
        uv run python robometer/evals/run_baseline_eval.py \
            "${COMMON_ARGS[@]}" \
            model_path="$model_path" \
            output_dir="$out_dir"
        echo "Done: $label"
    }
fi

# ---- completion check -------------------------------------------------------
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

# ---- dispatch ---------------------------------------------------------------
BASELINE_OUT="$OUTPUT_BASE/baseline_Robometer-4B"

if _all_qualities_done "$BASELINE_OUT"; then
    echo "  [skip] baseline_Robometer-4B (already done)"
else
    _run_model_eval "$BASELINE_MODEL" "$BASELINE_OUT" "baseline_Robometer-4B"
fi

echo ""
echo "Checkpoints:"
ls "$CHECKPOINT_BASE" | grep '^checkpoint-' | sort -V

for CKPT_DIR in "$CHECKPOINT_BASE"/checkpoint-*/; do
    CKPT_NAME=$(basename "$CKPT_DIR")
    CKPT_OUT="$OUTPUT_BASE/$CKPT_NAME"
    if _all_qualities_done "$CKPT_OUT"; then
        echo "  [skip] $CKPT_NAME"
        continue
    fi
    _run_model_eval "$CKPT_DIR" "$CKPT_OUT" "$CKPT_NAME"
done

echo ""
echo "=== Waiting for all model evaluations... ==="
_wait_all_slots
echo "=== All evals done. ==="

# ---- post-processing --------------------------------------------------------
echo ""
echo "=== Recomputing per-quality metrics... ==="
cd "$ROBOMETER_DIR"
uv run python "$RECOMPUTE_SCRIPT" --output-base "$OUTPUT_BASE"

echo ""
echo "=== Creating metric plots... ==="
uv run python "$COMBINE_SCRIPT" \
    --output-base "$OUTPUT_BASE" \
    --baseline-dir "$OUTPUT_BASE/baseline_Robometer-4B"

echo ""
echo "=== Done! ==="
echo "  Output  : $OUTPUT_BASE"
echo "  Logs    : $LOG_DIR/"
echo "  Plots   : $OUTPUT_BASE/metric_vs_step_*.png"
