# Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons

[![arXiv](https://img.shields.io/badge/arXiv-2603.02115-b31b1b.svg)](https://arxiv.org/abs/2603.02115)
[![GitHub](https://img.shields.io/badge/GitHub-robometer-181717?logo=github)](https://github.com/robometer/robometer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model](https://img.shields.io/badge/Model-FFD21E?logo=huggingface)](https://huggingface.co/robometer/Robometer-4B)
[![Dataset](https://img.shields.io/badge/Dataset-RBM--1M-FFD21E?logo=huggingface)](https://huggingface.co/datasets/)
[![RBM-1M Visualizer](https://img.shields.io/badge/Visualizer-RBM--FFD21E?logo=huggingface)](https://huggingface.co/spaces/rewardfm/visualizer)
[![RewardEval UI](https://img.shields.io/badge/%20RewardEval%20UI-FFD21E?logo=huggingface)](https://huggingface.co/spaces/rewardfm/rewardeval_ui)

<p align="center">
  <img src="assets/robometer.jpg" alt="Robometer" width="100%"/>
</p>

---

## Abstract

General-purpose robot reward models are typically trained to predict absolute task progress from expert demonstrations, providing only local, frame-level supervision. While effective for expert demonstrations, this paradigm scales poorly to large-scale robotics datasets where failed and suboptimal trajectories are abundant and assigning dense progress labels is ambiguous. We introduce **Robometer**, a scalable reward modeling framework that combines intra-trajectory progress supervision with inter-trajectory preference supervision. Robometer is trained with a dual objective: a frame-level progress loss that anchors reward magnitude on expert data, and a trajectory-comparison preference loss that imposes global ordering constraints across trajectories of the same task, enabling effective learning from both real and augmented failed trajectories. To support this formulation at scale, we curate **RBM-1M**, a reward-learning dataset comprising over one million trajectories spanning diverse robot embodiments and tasks, including substantial suboptimal and failure data. Across benchmarks and real-world evaluations, Robometer learns more generalizable reward functions than prior methods and improves robot learning performance across a diverse set of downstream applications.

---

## 📦 Package structure

```
robometer/
├── robometer/              # Main package
│   ├── data/               # Datasets and preprocessing
│   ├── configs/            # Hydra and experiment configs
│   ├── models/             # Model definitions
│   └── evals/              # Baseline evals (GVL, VLAC, Robodopamine, etc.)
├── evals/                  # Local eval & visualization scripts
│   ├── run_pred_visualization_no_labels.py          # Single-checkpoint visualization (no GT)
│   ├── run_baseline_eval_local_peg_insertion_vertical.py
│   ├── combine_checkpoint_eval_plots.py             # Multi-checkpoint combined plots (no GT)
│   └── combine_checkpoint_metric_plots.py           # Multi-checkpoint plots + metric summary
├── scripts/
│   ├── data/               # Dataset download, conversion, splitting
│   ├── inference/          # Inference example scripts
│   └── tools/              # Miscellaneous utilities
├── eval_commands/          # Shell scripts for evals
│   ├── reward_alignment.sh
│   ├── policy_ranking.sh
│   ├── confusion_matrix.sh
│   ├── sweep_rbm_checkpoints_no_labels.sh           # Sweep all rbm/ checkpoints (no GT)
│   └── sweep_rbm_checkpoints_with_metrics.sh        # Sweep all rbm/ checkpoints + metrics
├── train.py                # Training entrypoint
└── pyproject.toml          # Dependencies (uv)
```

---

## 🔬 Local Fine-tuning Pipeline (PegInsertionVertical-v1)

This section documents the end-to-end pipeline used for fine-tuning Robometer on locally collected peg-insertion trajectories.

### Step 1 — Split training and evaluation data

```bash
cd ~/RoboFAC/robometer
uv run python scripts/data/split_local_hf_dataset.py \
  --input /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset \
  --train-output /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_train \
  --eval-output /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_eval \
  --seed 42
```

Check data before training:
```bash
uv run python ../mani_envs/data_collection/visualize/visualize_random_progress_samples.py \
  --dataset /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_train \
  --num-samples 20 \
  --seed 0 \
  --output-dir ../mani_envs/data_collection/outputs/plots_train
```

### Step 2 — Data preprocessing

```bash
export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com

# Process train split
uv run python -m robometer.data.scripts.preprocess_local_hf_datasets \
  --config_path robometer/configs/preprocess_local_hf.yaml \
  --dataset_roots '["/data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_train/"]' \
  --dataset_names '["local/PegInsertionVertical_train"]'

# Process eval split
uv run python -m robometer.data.scripts.preprocess_local_hf_datasets \
  --config_path robometer/configs/preprocess_local_hf.yaml \
  --dataset_roots '["/data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_eval/"]' \
  --dataset_names '["local/PegInsertionVertical_eval"]'
```

Check processed data:
```bash
uv run python ../mani_envs/data_collection/visualize/visualize_processed_progress_samples.py \
  --dataset /data/yingxi/robometer/local_PegInsertionVertical_train \
  --num-samples 20 \
  --seed 0 \
  --output-dir ../mani_envs/data_collection/outputs/plots_cache_train
```

### Step 3 — Training

```bash
cd ~/RoboFAC/robometer

export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=8 \
  train.py \
  data.train_datasets=[local/PegInsertionVertical_train] \
  data.eval_datasets=[local/PegInsertionVertical_eval] \
  training.per_device_train_batch_size=8 \
  training.per_device_eval_batch_size=8 \
  training.max_steps=800 \
  data.max_frames=8 \
  data.resized_height=224 \
  data.resized_width=224 \
  training.save_strategy=steps \
  training.save_steps=50 \
  training.save_total_limit=8 \
  training.logging_steps=1 \
  logging.log_to=[wandb]
```

### Step 4 — Evaluation

**Evaluate pretrained Robometer-4B baseline:**
```bash
cd ~/RoboFAC/robometer
uv run python robometer/evals/run_baseline_eval_local_peg_insertion_vertical.py \
  reward_model=rbm \
  model_path=robometer/Robometer-4B \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=5 \
  custom_eval.reward_alignment_max_trajectories=30 \
  max_frames=8 \
  model_config.batch_size=16
```

**Compare oracle vs. fine-tuned checkpoint:**
```bash
export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com

uv run python robometer/evals/run_baseline_eval_local_peg_insertion_vertical.py \
  reward_model=rbm \
  model_path=/data/yingxi/robometer/logs/rbm/checkpoint-800/ \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=10 \
  custom_eval.reward_alignment_max_trajectories=30 \
  max_frames=8 \
  model_config.batch_size=16
```

**Evaluate with real-world data (no labels):**
```bash
export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com

uv run python robometer/evals/run_pred_visualization_no_labels.py \
  reward_model=rbm \
  model_path=/data/yingxi/robometer/logs/rbm/checkpoint-3700/ \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=10 \
  custom_eval.reward_alignment_max_trajectories=30 \
  max_frames=8 \
  model_config.batch_size=16
```

### Step 5 — Multi-checkpoint sweep & visualization

Evaluate **every** checkpoint under `logs/rbm/` in one shot and produce combined
per-trajectory plots overlaying all checkpoints against the Robometer-4B baseline.
Results are written to `/data/yingxi/robometer/all_checkpoint_eval_output*/`.

**Option A — No GT labels (visualization only)**

Runs `run_pred_visualization_no_labels.py` per checkpoint; combines predicted-progress
curves (Blues palette, light→dark) + Robometer-4B baseline (orange) into one PNG per
trajectory. Safe to interrupt and resume — already-evaluated checkpoints are skipped.

```bash
cd ~/RoboFAC/robometer
bash eval_commands/sweep_rbm_checkpoints_no_labels.sh 2>&1 | tee eval_sweep_no_labels.log
```

To re-run only the plotting step (results already on disk):
```bash
cd ~/RoboFAC/robometer
uv run python evals/combine_checkpoint_eval_plots.py \
  --output-base /data/yingxi/robometer/all_checkpoint_eval_output
```

Output: `all_checkpoint_eval_output/combined_trajectory_plots/<trajectory_id>.png`

---

**Option B — With GT labels (Pearson / Loss metrics)**

Runs `run_baseline_eval.py` per checkpoint (requires `target_progress` in dataset);
additionally produces a `metric_vs_training_step.png` summary figure with one subplot
per metric and the Robometer-4B baseline value as a horizontal dashed reference line.

```bash
cd ~/RoboFAC/robometer
bash eval_commands/sweep_rbm_checkpoints_with_metrics.sh 2>&1 | tee eval_sweep_metrics.log
```

To re-run only the plotting step:
```bash
cd ~/RoboFAC/robometer
uv run python evals/combine_checkpoint_metric_plots.py \
  --output-base /data/yingxi/robometer/all_checkpoint_eval_output_metrics \
  --baseline-dir /data/yingxi/robometer/all_checkpoint_eval_output_metrics/baseline_Robometer-4B
```

Output:
- `all_checkpoint_eval_output_metrics/combined_trajectory_plots/<trajectory_id>.png`
- `all_checkpoint_eval_output_metrics/metric_vs_training_step.png`

---

## 🛠️ Setup

### Prerequisites

- Git, Python 3.10+
- NVIDIA drivers (GPU)
- [uv](https://github.com/astral-sh/uv#installation) (recommended)

### Install (main env)

```bash
git clone https://github.com/aliang8/robometer.git
cd robometer

# Create venv and install
uv sync
```

### Dataset setup

```bash
hf auth
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/save/processed_datasets
./scripts/data/download_processed_datasets.sh
./scripts/data/untar_processed_datasets.sh
```

For raw download and preprocessing, see [📥 Download raw datasets](#-download-raw-datasets-optional) below.

---

## 🔍 Inference

Inference runs a **pretrained RBM model** on your own videos to get per-frame progress, per-frame success, and (for two trajectories) preference scores.

**Pretrained models (Hugging Face):**

- **[Robometer-4B](https://huggingface.co/robometer/Robometer-4B)** — general-purpose, trained on RBM-1M
- **[Robometer-LIBERO](https://huggingface.co/jesbu1/robometer-4b-fft-libero)** - fine-tuned on LIBERO-90, Object, Goal, Spatial, 10 + associated failure data.

### Inference via HTTP server

Start the eval server on your machine, then call it with a video and task:

```bash
uv run python robometer/evals/eval_server.py \
  server_url=0.0.0.0 \
  server_port=8000
```

Then run the client (no robometer dependency):

```bash
# SOAR
uv run python scripts/inference/example_inference.py \
  --eval-server-url http://localhost:8000 \
  --video scripts/example_videos/soar_put_green_stick_in_brown_bowl.mp4 \
  --task "Put green stick in brown bowl" \
  --fps 3

# Berkeley RPT (Wrist)
uv run python scripts/inference/example_inference.py \
  --eval-server-url http://localhost:8000 \
  --video scripts/example_videos/berkeley_rpt_stack_cup.mp4 \
  --task "Pick up the yellow cup and stack it on the other cup" \
  --fps 3

# Your own video
uv run python scripts/inference/example_inference.py \
  --eval-server-url http://localhost:8000 \
  --video /path/to/video.mp4 \
  --task "your task description"
```

To run the model locally (loads checkpoint from Hugging Face, no server):

```bash
uv run python scripts/inference/example_inference_local.py \
  --model-path robometer/Robometer-4B \
  --video /path/to/video.mp4 \
  --task "your task description"
```

---

## 🏋️ Training

### Train on RBM-1M

First, modify `robometer/configs/config.yaml`'s `wandb_entity` flag to your WandB entity. To disable WandB logging, remove "wandb" from the `log_to` list.

```bash
uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=N_GPUS_YOU_HAVE \
  train.py \
  data.train_datasets=[rbm-1m-id] \
  data.eval_datasets=[rbm-1m-ood] \
  data.max_frames=8 \
  model.train_progress_head=true \
  model.train_preference_head=true \
  training.max_steps=15000 \
  custom_eval.reward_alignment=[rbm-1m-ood] \
  custom_eval.policy_ranking=[rbm-1m-ood] \
  custom_eval.confusion_matrix=[rbm-1m-ood]
```

### Train on LIBERO

```bash
uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  train.py \
  data.train_datasets=[libero_pi0] \
  data.eval_datasets=[libero_pi0] \
  data.max_frames=8 \
  model.train_progress_head=true \
  model.train_preference_head=true \
  training.max_steps=5000 \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.policy_ranking=[libero_pi0]
```

See `robometer/configs/experiment_configs.py` for more options.

---

## 🔧 LoRA fine-tune Robometer for a new dataset

Full step-by-step: **[FINETUNE_ROBOMETER.md](FINETUNE_ROBOMETER.md)**.

- **Preprocessing:** Add your dataset to the preprocess config and run the preprocessor; for raw videos convert to RBM format first via `dataset_upload/`, then preprocess.
- **Fine-tuning:** Set `model.use_peft=true` and `training.load_from_checkpoint=robometer/Robometer-4B`, then train on your dataset.
- **Upload & inference:** Use `robometer/utils/upload_to_hub.py` to push checkpoints; run `scripts/inference/example_inference_local.py` with your Hub model.

---

## 📊 Evaluation (benchmarks)

### Reward alignment

```bash
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=robometer/Robometer-4B \
  custom_eval.eval_types=[reward_alignment] \
  custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=5 \
  custom_eval.reward_alignment_max_trajectories=30 \
  max_frames=8 \
  model_config.batch_size=32
```

### Policy ranking

```bash
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=robometer/Robometer-4B \
  custom_eval.eval_types=[policy_ranking] \
  custom_eval.policy_ranking=[rbm-1m-ood] \
  custom_eval.use_frame_steps=false \
  custom_eval.num_examples_per_quality_pr=1000 \
  max_frames=8 \
  model_config.batch_size=32
```

### Confusion matrix

```bash
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm \
  model_path=robometer/Robometer-4B \
  custom_eval.eval_types=[confusion_matrix] \
  custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top]] \
  max_frames=8 \
  model_config.batch_size=32
```

For baseline models (ReWIND, Robo-Dopamine, VLAC, RoboReward) see `eval_commands/*.sh`.

---

## 📊 Dataset generation

```bash
# AgiBotWorld
uv run python dataset_upload/generate_hf_dataset.py \
  --config_path=dataset_upload/configs/data_gen_configs/agibot_world.yaml

# LIBERO
uv run python dataset_upload/generate_hf_dataset.py \
  --config_path=dataset_upload/configs/data_gen.yaml \
  --dataset.dataset_path=LIBERO/libero/datasets/libero_90 \
  --dataset.dataset_name=libero_90
```

See `dataset_upload/README.md` and `dataset_upload/dataset_guides/` for adding new datasets.

---

## 📥 Download raw datasets (optional)

```bash
export ROBOMETER_DATASET_PATH=/path/to/your/robometer_dataset
./scripts/data/download_data.sh

# Preprocess
uv run python -m robometer.data.scripts.preprocess_datasets \
  --config robometer/configs/preprocess.yaml
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/save/processed_datasets
```

---

## 📑 License

This project is licensed under the [MIT License](LICENSE).

## BibTeX

```bibtex
@inproceedings{liang2026robometer,
  title     = {Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons},
  author    = {Anthony Liang and Yigit Korkmaz and Jiahui Zhang and Minyoung Hwang and Abrar Anwar and Sidhant Kaushik and Aditya Shah and Alex S. Huang and Luke Zettlemoyer and Dieter Fox and Yu Xiang and Anqi Li and Andreea Bobu and Abhishek Gupta and Stephen Tu and Erdem Biyik and Jesse Zhang},
  year      = {2026},
  booktitle = {Robotics: Science and Systems 2026},
}
```
