# Robometer 5-Task Collection

This note documents the training-data collection pipeline used by `robometer` for the 5-task setup driven by `mani_envs/data_collection/auto_collect_label`.

## Natural 5-Task Pilot Training

The finished natural-language 5-task pilot run is stored under:

```text
/data/yingxi/robometer/natural_five_task_20260921/training/natural_five_task_fsdp_pilot/
```

Run status:

- status: completed successfully
- exit file: `/data/yingxi/robometer/natural_five_task_20260921/logs/train_pilot_v2.exit` (`0`)
- log: `/data/yingxi/robometer/natural_five_task_20260921/logs/train_pilot_v2.log`
- trainer state: `/data/yingxi/robometer/natural_five_task_20260921/training/natural_five_task_fsdp_pilot/trainer_state.json`
- final step: `500 / 500`
- runtime: `44460.8298` seconds
- train loss: `0.14423589715361596`
- final checkpoint: `/data/yingxi/robometer/natural_five_task_20260921/training/natural_five_task_fsdp_pilot/checkpoint-500`
- checkpoints: `checkpoint-50`, `checkpoint-100`, ..., `checkpoint-500`

Key config:

- config: `/data/yingxi/robometer/natural_five_task_20260921/training/natural_five_task_fsdp_pilot/config.yaml`
- base model: `/data/yingxi/robometer/Qwen3-VL-4B-Instruct/`
- trainer: `rbm_heads`
- train dataset: `local/natural_five_task_train`
- max steps: `500`
- save interval: every `50` steps
- per-device train batch size: `2`
- gradient accumulation: `4`
- learning rate: `2e-05`, cosine schedule, warmup ratio `0.1`
- precision: `bf16`
- max frames: `8`
- progress loss: discrete, `10` bins
- trainable components: language model, progress head, preference head,
  success head
- frozen component: vision encoder

Check completion:

```bash
cat /data/yingxi/robometer/natural_five_task_20260921/logs/train_pilot_v2.exit
tail -80 /data/yingxi/robometer/natural_five_task_20260921/logs/train_pilot_v2.log
python3 - <<'PY'
import json
p = "/data/yingxi/robometer/natural_five_task_20260921/training/natural_five_task_fsdp_pilot/trainer_state.json"
with open(p) as f:
    state = json.load(f)
print(state["global_step"], state["max_steps"])
print(state["log_history"][-1])
PY
```

## Natural 5-Task Checkpoint Eval

Use `evals/eval_natural_five_task.py` for the natural-language 5-task pilot
checkpoints. It reports three metric families in one run:

- `progress`: per-frame MAE and Spearman over the sampled frames.
- `failure_detection`: progress-head based failure classification metrics.
  The default fixed-threshold rule is
  `failure if final_pred_progress < 0.5`; `success_prob` / success head is not
  used for the reported failure detection metrics.
- `filtering`: top-k successful-trajectory rate for `pearson`, `final`,
  `delta`, `slope`, and `late_mean_delta` scores.

Run from the robometer repo:

```bash
cd /data/yingxi/RoboFPE/robometer
export CUDA_VISIBLE_DEVICES=0
```

Smoke test, using the small smoke checkpoint and smoke dataset:

```bash
.venv/bin/python evals/eval_natural_five_task.py smoke --force
```

Single formal checkpoint:

```bash
.venv/bin/python evals/eval_natural_five_task.py single \
  --checkpoint /data/yingxi/robometer/natural_five_task_20260921/training/natural_five_task_fsdp_pilot/checkpoint-500
```

Full formal sweep over all `checkpoint-*` directories, sequentially:

```bash
.venv/bin/python evals/eval_natural_five_task.py sweep
```

Background formal sweep:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 .venv/bin/python evals/eval_natural_five_task.py sweep \
  > /data/yingxi/robometer/natural_five_task_20260921/logs/eval_sweep.log 2>&1 &
```

Default formal inputs:

- dataset: `/data/yingxi/robometer/natural_five_task_20260921/processed/local_natural_five_task_test/processed_dataset`
- manifest: `/data/yingxi/robometer/natural_five_task_20260921/local_hf/natural_five_task_test/selection_manifest.json`
- checkpoints: `/data/yingxi/robometer/natural_five_task_20260921/training/natural_five_task_fsdp_pilot/checkpoint-*`
- output: `/data/yingxi/robometer/natural_five_task_20260921/eval_sweep`

Outputs:

- `run_config.json`: dataset, manifest hash, checkpoint list, thresholds, and code commit.
- `step_000XXX_metrics.json`: metrics for one checkpoint.
- `step_000XXX_records.json`: per-trajectory predictions used to compute metrics.
- `metrics.json`: concatenated metrics for all evaluated checkpoints.
- `sweep_summary.json`: compact per-checkpoint summary.
- `per_task_step500_summary.csv`: final-checkpoint per-task metrics.
- `plots/*.png` and `plots/*.jpg`: image exports for reports.
- `plots/progress_success_failure_average_task_macro.csv`: progress split
  curves for success, failure, and average.
- `plots/per_task_dashboards/`: one PNG/JPG dashboard per task, plus
  `per_task_dashboard_timeseries.csv`.

Generated plots:

![Progress metrics vs checkpoint step](/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/progress_metrics_vs_step.png)

![Progress success metrics vs checkpoint step](/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/progress_success_metrics_vs_step.png)

![Progress failure metrics vs checkpoint step](/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/progress_failure_metrics_vs_step.png)

![Progress average metrics vs checkpoint step](/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/progress_average_metrics_vs_step.png)

![Failure detection metrics vs checkpoint step](/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/failure_metrics_vs_step.png)

![Filtering best rate vs checkpoint step](/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/filtering_best_rate_vs_step.png)

Per-task dashboards:

- `/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/per_task_dashboards/PickCube-ball_dashboard.png`
- `/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/per_task_dashboards/PlugCharger-v1_dashboard.png`
- `/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/per_task_dashboards/PullCube-block_dashboard.png`
- `/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/per_task_dashboards/PushCube-v1_dashboard.png`
- `/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/plots/per_task_dashboards/StackCube-v1_dashboard.png`

The previous success-head failure-detection metrics are preserved under:

```text
/data/yingxi/robometer/natural_five_task_20260921/eval_sweep/success_prob_failure_detection_backup/
```

Existing step metrics are skipped by default so interrupted sweeps can resume.
Pass `--force` to recompute.

## StackCube Failure Comparison: checkpoint-400 vs basefixed

The StackCube failure test set can be evaluated against both the trained
`checkpoint-400` and the pre-training `robometer-4b_basefixed` checkpoint:

```bash
cd /data/yingxi/RoboFPE/robometer
export CUDA_VISIBLE_DEVICES=0

# Smoke test: two StackCube failure trajectories
.venv/bin/python evals/eval_stackcube_failure_compare.py \
  --limit 2 \
  --output-dir /data/yingxi/robometer/natural_five_task_20260921/eval_stackcube_failure_smoke

# Formal test: all 50 StackCube failure trajectories
.venv/bin/python evals/eval_stackcube_failure_compare.py \
  --output-dir /data/yingxi/robometer/natural_five_task_20260921/eval_stackcube_failure_ckpt400_vs_basefixed
```

The script evaluates:

- trained model:
  `/data/yingxi/robometer/natural_five_task_20260921/training/natural_five_task_fsdp_pilot/checkpoint-400`
- base model:
  `/data/yingxi/robometer/robometer-4b_basefixed`
- dataset:
  `/data/yingxi/robometer/natural_five_task_20260921/local_hf/natural_five_task_test`

Outputs are written under `per_trajectory/<trajectory-id>/`:

- `render_camera_frames/`: all 32 preprocessed label/render-camera frames.
- `dashboard.png` and `dashboard.jpg`: downsampled render-camera frames above
  the label, checkpoint-400, and basefixed progress curves.
- `records.json`: raw predictions, labels, frame indices, and image paths.
- `metrics.json`: mean per-trajectory MAE and Spearman for both models.
- `summary.csv`: one metric row per trajectory.

Frame alignment is explicit. Each source failure trajectory has 300 progress
labels and 301 render-camera frames. The first video frame is the reset frame,
so preprocessing uses:

```text
progress_indices = floor(i * 300 / 32)
render_indices = progress_indices + video_frame_offset
```

The full 32 aligned points are saved as images. Model inference uses the same
8-frame subsampling as the five-task eval; the dashboard uses those same eight
render-camera timesteps on its x-axis, so every image column is aligned with
the corresponding curve point.

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
