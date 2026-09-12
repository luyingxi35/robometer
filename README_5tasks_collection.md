# Robometer 5-Task Collection

This note documents the training-data collection pipeline used by `robometer` for the 5-task setup driven by `mani_envs/data_collection/auto_collect_label`.

## Tasks

The 5 tasks currently used by this collection flow are:
- `PegInsertionVertical-v1`
- `PegInsertionSide-v1`
- `PlugCharger-v1`
- `PushCube-v1`
- `StackCube-v1`

## Collection command

Use this wrapper:

```bash
bash /data/yingxi/RoboFPE/robometer/run_5task_collection.sh
```

The wrapper uses `/data/yingxi/robometer/failure_detection_env/bin/python` by default, so activating a conda environment is not required. CUDA shared libraries are loaded from `/data/yingxi/robofac` by the launcher.

## Runtime paths

Typical values:
- repo root: `/data/yingxi/RoboFPE`
- data root: `/data/yingxi/robometer/failure_detection_5ind_10ood`
- raw success input: `${DATA_ROOT}/raw/<task>/success`
- raw failure input: `${DATA_ROOT}/raw/<task>/failure`
- merged labeled output: `${DATA_ROOT}/<task>/hf_dataset`

The launcher `run_auto_collect_label.sh` forwards these to `auto_collect_label.py` and accepts:
- `TASK_ID`
- `NUM_SUCCESS_DEMO`
- `NUM_FAILURE_DEMO_EACH_TYPE`
- `LABEL_METHOD`
- `SIMILARITY_MODE`
- `WORK_DIR`
- `SUCCESS_DIR`
- `FAILURE_DIR`

The loop calls:

```bash
bash mani_envs/data_collection/auto_collect_label/run_auto_collect_label.sh --skip-query
```

for each task.

Override as needed:

```bash
CUDA_VISIBLE_DEVICES=1 NUM_SUCCESS_DEMO=20 NUM_FAILURE_DEMO_EACH_TYPE=5 \
  bash /data/yingxi/RoboFPE/robometer/run_5task_collection.sh
```

## Smoke test

Run one task with one success and one failure demo:

```bash
TASKS="PegInsertionVertical-v1" \
NUM_SUCCESS_DEMO=1 \
NUM_FAILURE_DEMO_EACH_TYPE=1 \
DATA_ROOT=/tmp/robometer_smoke \
  bash /data/yingxi/RoboFPE/robometer/run_5task_collection.sh
```

The command must exit with code `0` and create:

```text
/tmp/robometer_smoke/PegInsertionVertical-v1/hf_dataset/
```

On the inspected server, the wrapper reached schema generation but SAPIEN crashed during GPU environment initialization with exit code `139`; therefore the collection smoke test is currently blocked by the server's SAPIEN/Vulkan runtime, before H5/MP4 collection starts.

For a fuller GPU/runtime diagnosis, run:

```bash
bash /data/yingxi/RoboFPE/robometer/gpu_smoke_test.sh
```

This checks, in order:
- H100 visibility with `nvidia-smi`
- a real Torch CUDA matrix multiplication and synchronization
- `sapien.Device("cuda:0")`
- `PegInsertionVertical-v1` creation/reset with `sim_backend="gpu"`
- one real `run_success.py` trajectory with MP4 recording

The test continues after a crash in an individual subprocess and returns nonzero if any layer fails. On the current server, Torch CUDA passes, while SAPIEN CUDA initialization and the real trajectory test exit with `139`.

## What gets saved

Per task, the pipeline writes:
- `auto_config/task_schema.json`
- `auto_config/env_code_parsed.json`
- `auto_config/failure_collection_report.json`
- `successful_labeld/`
- `failure_labeled/`
- `suboptimal_labeled/`
- `hf_dataset/`
- `video_copy_manifest.json`

Raw episodes are stored as:
- `*.h5`
- `*.json`
- `*.mp4`

The raw H5 files hold:
- `actions`
- `env_states`
- `success`
- optional `rewards`

## Final HF row schema

The labeled HuggingFace dataset uses these top-level fields:
- `id`
- `task`
- `data_source`
- `quality_label`
- `is_robot`
- `frames_video`
- `frames`
- `target_progress`
- `relative_pose`
- `way_points`
- `partial_success`
- `actions`
- `states`
- `metadata`
- `lang_vector`

Important metadata fields:
- `success`
- `episode_return`
- `source_h5`
- `source_traj_key`
- `source_episode_idx`
- `stage_records`
- `waypoint_timesteps`
- `stage_end_timesteps`
- task-specific auto-label traces

`frames` points to the MP4 path. `lang_vector` can be `null` in the raw labeled cache and is filled during preprocessing.

## Example row

Example from the current on-disk collection:

```json
{
  "id": "PegInsertionVertical-v1_success_0",
  "task": "Coordinate with the wrist-camera-guided robot arm to take the target peg, and lower it vertically inside the hole.",
  "data_source": "gen_progress_success",
  "quality_label": "successful_labeled",
  "is_robot": true,
  "frames": "/data/yingxi/robometer/failure_detection_5ind_10ood/PegInsertionVertical-v1/successful_labeld/videos/PegInsertionVertical-v1_success_0.mp4",
  "partial_success": 1.0
}
```

## Current statistics

Current `hf_dataset` row counts on disk:

| Task | Rows | Successful | Failure | Suboptimal |
|---|---:|---:|---:|---:|
| PegInsertionVertical-v1 | 260 | 30 | 112 | 118 |
| PegInsertionSide-v1 | 230 | 30 | 103 | 97 |
| PlugCharger-v1 | 230 | 30 | 94 | 106 |
| PushCube-v1 | 110 | 30 | 20 | 60 |
| StackCube-v1 | 220 | 30 | 84 | 106 |

## SFT Collection Camera Update

The separate SFT collector lives in:

```text
/data/yingxi/RLinf_RoboFAPE/run_train/robofpe_sft_data/collect_sft_data.py
```

Its target runtime remains:

```text
/data/yingxi/robometer/failure_detection_env/bin/python
```

For the four IND tasks other than `PushCube-v1`, collect 3200 successful
trajectories per task with 16 independent single-environment workers across
four GPUs. The SFT policy remains a two-image policy:

- `observation.images.top` contains human `render_camera` pixels.
- `observation.images.wrist` contains `hand_camera` pixels.

The `top` field name is retained for OpenPI compatibility. Raw H5 files may
still contain `base_camera` observations, and raw human-render MP4 files are
kept for inspection, but `base_camera` is not used as the new SFT main image.

Run the four task commands sequentially from
`/data/yingxi/RLinf_RoboFAPE`:

```bash
export PYTHON_BIN=/data/yingxi/robometer/failure_detection_env/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_PYTHON_LIB_ROOT=/data/yingxi/robofac/lib/python3.10/site-packages/nvidia
export CUDA_LIB_DIRS="$(find "$CUDA_PYTHON_LIB_ROOT" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')"
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MS_ASSET_DIR=/data/yingxi/robofac

"$PYTHON_BIN" run_train/robofpe_sft_data/collect_sft_data.py collect \
  --task-id PegInsertionVertical-v1 --num-traj 3200 \
  --output-dir /data/yingxi/datasets/robofpe_sft/PegInsertionVertical-v1_render_wrist \
  --robot-uids panda_wristcam --success-only --randomize-initial-poses \
  --randomize-wrist-camera --randomize-render-camera --randomize-lighting \
  --save-video --sim-backend gpu --num-workers 16 --gpu-ids 0,1,2,3 \
  --max-attempts-per-traj 100 --seed 20260902

"$PYTHON_BIN" run_train/robofpe_sft_data/collect_sft_data.py collect \
  --task-id PegInsertionSide-v1 --num-traj 3200 \
  --output-dir /data/yingxi/datasets/robofpe_sft/PegInsertionSide-v1_render_wrist \
  --robot-uids panda_wristcam --success-only --randomize-initial-poses \
  --randomize-wrist-camera --randomize-render-camera --randomize-lighting \
  --save-video --sim-backend gpu --num-workers 16 --gpu-ids 0,1,2,3 \
  --max-attempts-per-traj 100 --seed 30260902

"$PYTHON_BIN" run_train/robofpe_sft_data/collect_sft_data.py collect \
  --task-id PlugCharger-v1 --num-traj 3200 \
  --output-dir /data/yingxi/datasets/robofpe_sft/PlugCharger-v1_render_wrist \
  --robot-uids panda_wristcam --success-only --randomize-initial-poses \
  --randomize-wrist-camera --randomize-render-camera --randomize-lighting \
  --save-video --sim-backend gpu --num-workers 16 --gpu-ids 0,1,2,3 \
  --max-attempts-per-traj 100 --seed 40260902

"$PYTHON_BIN" run_train/robofpe_sft_data/collect_sft_data.py collect \
  --task-id StackCube-v1 --num-traj 3200 \
  --output-dir /data/yingxi/datasets/robofpe_sft/StackCube-v1_render_wrist \
  --robot-uids panda_wristcam --success-only --randomize-initial-poses \
  --randomize-wrist-camera --randomize-render-camera --randomize-lighting \
  --save-video --sim-backend gpu --num-workers 16 --gpu-ids 0,1,2,3 \
  --max-attempts-per-traj 100 --seed 50260902
```

The SFT converter writes two 224x224 training videos per episode and records
the camera mapping in `lerobot/meta/info.json`. The existing OpenPI configs
continue to use `num_images_in_input: 2`.

Future RL camera migration is not implemented in this update. The intended
future change is `render_camera -> main_images` and
`hand_camera -> wrist_images`, still with two policy image inputs.

## Training handoff

Training does not read the raw H5 files directly. The flow is:

1. collect raw success/failure episodes
2. label them into `successful_labeld`, `failure_labeled`, `suboptimal_labeled`
3. use the task-level `hf_dataset`
4. preprocess that HuggingFace dataset into the cache used by `train.py`

For the Robometer fine-tuning path, this collection step feeds into `robometer/README.md` and `robometer/FINETUNE_ROBOMETER.md`.

## GPU Environment Recovery (2026-09-02)

The target runtime for this collection is:

```text
/data/yingxi/robometer/failure_detection_env/bin/python
```

Do not install or switch the collection to `/root/failure_detection_env`. The
GPU smoke test was run from `/data/yingxi/RoboFPE` with:

```bash
cd /data/yingxi/RoboFPE
bash robometer/gpu_smoke_test.sh
```

### Failure sequence and fixes

1. H100 visibility, the Torch CUDA kernel, and `sapien.Device("cuda:0")`
   already passed.
2. The first Python failure was a missing `httpx` dependency for
   `huggingface_hub`. `httpx==0.28.1` and its HTTP dependencies were installed
   into `failure_detection_env`.
3. ManiSkill import then required `arm-pytorch-utilities==0.5.0` and several
   normal runtime dependencies, including `cycler`, `contourpy`, `fonttools`,
   `dacite`, `defusedxml`, `GitPython`, `IPython`, `click`, `cloudpickle`, and
   `absl-py`.
4. The repository tasks use the locally extended ManiSkill classes
   `noTableSceneBuilder` and `noTableSceneBuilder_microwave`. These classes were
   present in the existing `/data/yingxi/robofac` ManiSkill installation but
   missing from the target environment. The patched `table` scene-builder files
   were copied into the target environment, with a backup kept beside them.
5. `mplib.Planner` crashed with `SIGSEGV` while the target environment used
   NumPy 2.2.6. Aligning the environment to the known working versions
   (`numpy==1.26.4`, `opencv-python==4.11.0.86`, and `triton==3.4.0`) fixed the
   native planner crash.
6. MP4 writing then failed because ImageIO had no video backend. Installing
   `imageio-ffmpeg==0.6.0` fixed video encoding.

### Verification

The final smoke test passed all stages:

- H100 detection
- Torch CUDA matrix multiplication
- SAPIEN CUDA device creation
- GPU ManiSkill environment creation, reset, and close
- one successful `PegInsertionVertical-v1` motion-planning trajectory
- MP4, H5, and JSON output

The final output was written to:

```text
/tmp/robometer_gpu_smoke_20260902_160711/
```

The trajectory completed with success rate `1` and episode length `188`.
The collection launcher already forwards `PYTHON_BIN`, sets the required CUDA
library path, and uses the target environment by default; no simulation
backend change was required. It now also checks that the selected Python
executable exists before starting collection.

The remaining `pip check` messages concern Torch's NVIDIA package metadata.
The CUDA libraries are intentionally provided from the existing
`/data/yingxi/robofac` installation through `LD_LIBRARY_PATH`; the runtime
smoke test passed with this arrangement.
