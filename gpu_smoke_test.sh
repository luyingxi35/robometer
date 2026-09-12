#!/usr/bin/env bash
set +e

export CUDA_VISIBLE_DEVICES=0
export MS_ASSET_DIR=/data/yingxi/robofac
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

PY=${PYTHON_BIN:-/data/yingxi/robometer/failure_detection_env/bin/python}
ROOT=/data/yingxi/RoboFPE
OUT=/tmp/robometer_gpu_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
FAILED=0
CUDA_PYTHON_LIB_ROOT=${CUDA_PYTHON_LIB_ROOT:-/data/yingxi/robofac/lib/python3.10/site-packages/nvidia}
if [[ -d "$CUDA_PYTHON_LIB_ROOT" ]]; then
  CUDA_LIB_DIRS=$(find "$CUDA_PYTHON_LIB_ROOT" -mindepth 2 -maxdepth 2 -type d -name lib -printf "%p:")
  export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$ROOT:$ROOT/mani_envs:$ROOT/mani_envs/data_collection:$ROOT/mani_envs/data_collection/collect:$ROOT/mani_envs/data_collection/utils:${PYTHONPATH:-}"

echo "=== GPU INFO ==="
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader

echo "=== TORCH CUDA KERNEL ==="
"$PY" - <<'PY'
import torch
x = torch.randn((4096, 4096), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("torch_cuda_ok", torch.cuda.get_device_name(0), float(y.mean()))
PY
S=$?
echo "TORCH_EXIT=$S"
[[ "$S" -eq 0 ]] || FAILED=1

echo "=== SAPIEN CUDA DEVICE ==="
"$PY" - <<'PY'
import sapien
print("before_device", flush=True)
d = sapien.Device("cuda:0")
print("sapien_device_ok", d, flush=True)
PY
S=$?
echo "SAPIEN_DEVICE_EXIT=$S"
[[ "$S" -eq 0 ]] || FAILED=1

echo "=== MANISKILL ENV GPU ==="
cd "$ROOT"
"$PY" - <<'PY'
import gymnasium as gym
import tasks
print("before_env", flush=True)
env = gym.make(
    "PegInsertionVertical-v1",
    obs_mode="none",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    sim_backend="gpu",
)
print("env_created", flush=True)
env.reset(seed=0)
print("env_reset", flush=True)
env.close()
print("env_closed", flush=True)
PY
S=$?
echo "MANISKILL_EXIT=$S"
[[ "$S" -eq 0 ]] || FAILED=1

echo "=== RUN SUCCESS TRAJECTORY ==="
cd "$ROOT/mani_envs/data_collection/run"
timeout 90 "$PY" -X faulthandler run_success.py \
  --env-id PegInsertionVertical-v1 \
  --num-traj 1 \
  --traj-name gpu_smoke \
  --record-dir "$OUT" \
  --num-procs 1 \
  --shader default \
  --sim-backend gpu \
  --render-mode rgb_array \
  --obs-mode none \
  --only-count-success \
  --save-video \
  --max-episode-steps 250
S=$?
echo "RUN_SUCCESS_EXIT=$S"
[[ "$S" -eq 0 ]] || FAILED=1

echo "=== OUTPUTS ==="
find "$OUT" -type f \( -name "*.h5" -o -name "*.mp4" -o -name "*.json" \) -print | sort | sed -n "1,80p"
echo "OUT=$OUT"
exit "$FAILED"
