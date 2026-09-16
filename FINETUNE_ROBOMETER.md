# Fine-tuning Robometer on Your Own Data

Preprocess a dataset, LoRA fine-tune from Robometer-4B, upload to the Hub, run inference. Example: [MINT-SJTU/RoboFAC-dataset](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset).

---

## 1. Preprocessing (RoboFAC)

Training needs a **preprocessed** cache from a HuggingFace dataset in RBM format. RoboFAC is folder-based; convert first, then preprocess.

1. **Download** (into `ROBOMETER_DATASET_PATH`):
   ```bash
   export ROBOMETER_DATASET_PATH=/path/to/your/robometer_dataset
   huggingface-cli download MINT-SJTU/RoboFAC-dataset --local-dir $ROBOMETER_DATASET_PATH/RoboFAC-dataset
   ```

2. **Convert and push to Hub** (required for recommended flow):
   ```bash
   export HF_TOKEN=your_token_here

   uv run python -m dataset_upload.generate_hf_dataset \
     --config_path dataset_upload/configs/data_gen_configs/robofac.yaml \
     --dataset.dataset_path=$ROBOMETER_DATASET_PATH/RoboFAC-dataset \
     --hub.push_to_hub=true \
     --hub.hub_repo_id=robofac_rbm
   ```

3. **If you pushed:** download converted repo and set preprocess config:
   ```bash
   huggingface-cli download aliangdw/robofac_rbm --local-dir $ROBOMETER_DATASET_PATH/robofac_rbm
   ```
   In `preprocess_finetune.yaml`: `train_datasets: ["aliangdw/robofac_rbm"]`, `train_subsets: [["robofac"]]`.

4. **Preprocess:**
   ```bash
   export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/save/processed_datasets

   uv run python -m robometer.data.scripts.preprocess_datasets \
     --config robometer/configs/preprocess_finetune.yaml \
     --cache_dir=$ROBOMETER_PROCESSED_DATASETS_PATH
   ```
   Use the same path for training.

**Other datasets:** RBM-style Hub datasets: set `train_datasets`/`train_subsets` and `ROBOMETER_DATASET_PATH` in the preprocess config, then run step 4. Raw data: add a loader (see [CustomDataset.md](dataset_upload/dataset_guides/CustomDataset.md)).


## IMPORTANT STEP: Add a success / progress cutoff to `dataset_success_cutoff.txt`

Add a success / progress cutoff to `dataset_success_cutoff.txt` for your dataset.
This is used to threshold progress and success for each trajectory.

For example, if your trajectories have 10 frames and most of them are successful by frame 9, then you should set the cutoff to 0.9.
This is very important to ensure well-calibrated progress and success predictions.
We need this because most teleoperated datasets don't have consistent trajectory endpoints, and computing linear progress and success labels based on assuming the endpoint is at the last frame is not reliable.

Modify: `robometer/data/dataset_success_cutoff.txt` to add your cutoff corresponding to the name of the processed `data_source` in the preprocessing config you made in the above steps.

### Again, *VERY IMPORTANT* to ensure well-calibrated progress and success predictions.

If your dataset is simulation, it's most likely that you will have a dataset cutoff of 1.0 because you can get perfect trajectory ends in simulation.

## 2. LoRA fine-tuning

Use PEFT (LoRA) and `load_from_checkpoint` from a Qwen3-4B–based RBM checkpoint.

```bash
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/your/processed_datasets

uv run python train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true \
  model.train_preference_head=true \
  data.train_datasets=[aliangdw_robofac_rbm_robofac] \
  data.eval_datasets=[aliangdw_robofac_rbm_robofac] \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.warmup_ratio=0.1 \
  training.weight_decay=0.01 \
  training.max_steps=1000 \
  training.output_dir=./logs \
  training.exp_name=robometer4b_lora_robofac_2 \
  logging.log_to=[wandb] \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=[aliangdw_robofac_rbm_robofac] \
  custom_eval.policy_ranking=[aliangdw_robofac_rbm_robofac] \
  logging.save_best.metric_names=[eval_rew_align/pearson_robofac,eval_p_rank/kendall_last_robofac] \
  logging.save_best.greater_is_better=[true,true] \
  training.overwrite_output_dir=True \
  training.eval_steps=50 \
  training.custom_eval_steps=50
```

The short name `robofac` is defined in `name_mapping.py` for `aliangdw_robofac_rbm_robofac`.

**Tunable LoRA / training:** Override `training.learning_rate`, `training.warmup_ratio`, `training.weight_decay`, `training.gradient_accumulation_steps`, or `training.max_steps` as needed. Defaults above match `robometer/configs/config.yaml`. Multi-GPU: `uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml train.py ...` (same overrides).

### Full fine-tuning (no PEFT)

Load the same checkpoint but train the full model (no LoRA). Uses more memory; lower `per_device_train_batch_size` or gradient accumulation if needed.

```bash
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/your/processed_datasets

uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=N_GPUS_YOU_HAVE train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=false \
  model.train_progress_head=true \
  model.train_preference_head=true \
  data.train_datasets=[aliangdw_robofac_rbm_robofac] \
  data.eval_datasets=[aliangdw_robofac_rbm_robofac] \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.warmup_ratio=0.1 \
  training.weight_decay=0.01 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=500 \
  training.output_dir=./logs \
  training.exp_name=robometer4b_full_robofac \
  logging.log_to=[wandb] \
  custom_eval.reward_alignment=[aliangdw_robofac_rbm_robofac] \
  custom_eval.policy_ranking=[aliangdw_robofac_rbm_robofac] \
  logging.save_best.metric_names=[eval_rew_align/pearson_robofac,eval_p_rank/kendall_last_robofac] \
  logging.save_best.greater_is_better=[true,true] \
  training.eval_steps=50 \
  training.custom_eval_steps=50
```


---

## 3. Upload to Hub

```bash
uv run python robometer/utils/upload_to_hub.py \
  --model_dir ./logs/robometer4b_lora_robofac/checkpoint-500 \
  --hub_model_id aliangdw/robometer-4b-lora-robofac \
  --base_model "Qwen/Qwen3-VL-4B-Instruct" \
  --commit_message "LoRA fine-tune on RoboFAC"
```

Or enable `logging.save_best.upload_to_hub: true` in config for upload during training.

---

## 4. Inference

```bash
uv run python scripts/example_inference_local.py \
  --model-path aliangdw/robometer-4b-lora-robofac \
  --video /path/to/video.mp4 \
  --task "Insert the cylinder"
```

Server: `uv run python robometer/evals/eval_server.py ... model_path=aliangdw/robometer-4b-lora-robofac`. Eval: `run_baseline_eval.py` with `reward_model=rbm`, `model_path=...` (see [README](README.md)).

---

## 5. Baseline: Fine-tune from base Qwen-VL (no Robometer checkpoint)

For comparison, run the same `train.py` on the same data but **without** loading a Robometer checkpoint. Training starts from the base Qwen-VL plus randomly initialized progress/preference heads.

```bash
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/your/processed_datasets

uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=N_GPUS_YOU_HAVE train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true \
  model.train_preference_head=true \
  data.train_datasets=[aliangdw_robofac_rbm_robofac] \
  data.eval_datasets=[aliangdw_robofac_rbm_robofac] \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.warmup_ratio=0.1 \
  training.weight_decay=0.01 \
  training.max_steps=1000 \
  training.output_dir=./logs \
  training.exp_name=qwen3vl_lora_robofac_baseline \
  logging.log_to=[wandb] \
  custom_eval.reward_alignment=[aliangdw_robofac_rbm_robofac] \
  custom_eval.policy_ranking=[aliangdw_robofac_rbm_robofac] \
  logging.save_best.metric_names=[eval_rew_align/pearson_robofac,eval_p_rank/kendall_last_robofac] \
  logging.save_best.greater_is_better=[true,true] \
  training.eval_steps=50 \
  training.custom_eval_steps=50

uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml --num_processes=N_GPUS_YOU_HAVE train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=false \
  model.train_progress_head=true \
  model.train_preference_head=true \
  data.train_datasets=[aliangdw_robofac_rbm_robofac] \
  data.eval_datasets=[aliangdw_robofac_rbm_robofac] \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.warmup_ratio=0.1 \
  training.weight_decay=0.01 \
  training.max_steps=1000 \
  training.output_dir=./logs \
  training.exp_name=qwen3vl_robofac_baseline \
  logging.log_to=[wandb] \
  custom_eval.reward_alignment=[aliangdw_robofac_rbm_robofac] \
  custom_eval.policy_ranking=[aliangdw_robofac_rbm_robofac] \
  logging.save_best.metric_names=[eval_rew_align/pearson_robofac,eval_p_rank/kendall_last_robofac] \
  logging.save_best.greater_is_better=[true,true] \
  training.eval_steps=50 \
  training.custom_eval_steps=50
```

---

## 6. Natural Dual 500-Step Checkpoint Sweep Runbook

This is the validated full-fine-tuning setup for the natural StackCube/PushCube
dual-task training cache. It starts from the fixed RoboMeter base checkpoint and
is intended to produce a dense, reproducible checkpoint sweep.

```bash
cd /data/yingxi/RoboFPE/robometer
export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer/natural_progress_collection_20260914_4gpu/robometer_train_processed
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes 4 --main_process_port 29542 train.py \
  --config-path /data/yingxi/robometer/natural_progress_collection_20260914_4gpu/robometer_train_full_logs_b4/natural_dual_fsdp_full_20k_b4 \
  --config-name config \
  training.exp_name=natural_dual_fsdp_full_500_save50 \
  training.output_dir=/data/yingxi/robometer/natural_progress_collection_20260914_4gpu/robometer_train_save50_logs_b4 \
  training.max_steps=500 \
  training.save_steps=50 \
  training.save_total_limit=11 \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=4 \
  logging.save_best.save_every=50
```

Expected checkpoints are `checkpoint-50`, `checkpoint-100`, ..., and
`checkpoint-500` under:

`/data/yingxi/robometer/natural_progress_collection_20260914_4gpu/robometer_train_save50_logs_b4/natural_dual_fsdp_full_500_save50`

The base model is the step-0 evaluation point. Do not run the checkpoint sweep
until all ten saved checkpoints contain `config.json` and their model shard
files. `save_total_limit` must be at least 11: a smaller value silently deletes
older checkpoints and leaves the metrics-vs-step plot with missing points.

### OOM incident and recovery

The first four-rank attempt used `per_device_train_batch_size=4` and failed at
the first backward pass with `torch.OutOfMemoryError`: each 80 GB GPU had about
5.76 GB free and backward requested a further 9.22 GB. This is activation and
allocator pressure, not an FSDP or checkpoint-saving failure.

The recovery configuration above halves the per-rank microbatch to 2 while
keeping gradient accumulation at 4. With four ranks, its effective global batch
is `4 * 2 * 4 = 32`, matching the prior successful run's effective batch. The
allocator setting enables expandable segments to reduce fragmentation. Verify
the first real optimization step and `checkpoint-50` before treating the run as
stable.

The launcher must explicitly export `ROBOMETER_PROCESSED_DATASETS_PATH`; a
background shell does not necessarily inherit it. Without that variable,
training exits before data setup with `ValueError: ROBOMETER_PROCESSED_DATASETS_PATH not set`.

## 7. Labeled Progress Regression and Control-Group Semantics

The canonical labeled quality names are `successful_labeled`,
`suboptimal_labeled`, and `failure_labeled`. Each labeled trajectory carries
`target_progress`, and preference samples must use that target for the
preference-path progress loss when `predict_pref_progress=true`; a direct
progress sample is not required for this supervision.

The progress mask regression was introduced by RoboMeter commit `7429e3e`,
which removed the labeled-quality branch from `should_compute_progress()`.
With that branch absent, a preference batch can still report
`total_preferences=1` while masking every labeled trajectory's progress target,
so the progress head receives no useful gradient. `total_progress=0` alone is
not proof of this failure because `[1, 0, 0]` intentionally selects only the
preference pathway; inspect the effective progress mask and `pref_prog_loss`.

The fix restores progress supervision for all three canonical `*_labeled`
labels while preserving the existing behavior for ordinary `successful`,
rewind, failure, and preference-only samples. Do not use `success_labeled` as a
new spelling; it is not a recognized label.

For the motion-plan success control group:

- `successful_labeled` uses dataset `target_progress` and disables labeled
  trajectory rewind augmentation (`data.labeled_progress_disable_rewind=true`).
- `successful` uses ordinary success semantics and keeps rewind augmentation
  enabled (`data.labeled_progress_disable_rewind=false`).

Before a full run, execute the labeled progress unit tests and a short train
smoke test. The smoke log must contain a nonzero `pref_prog_loss`; preference
counts alone do not establish that progress supervision is active.

## 8. Frozen Natural-Dual Checkpoint Sweep Evaluation

The sweep evaluates the frozen exact render-camera test set for StackCube-v1
and PushCube-v1. It reports per-quality progress MSE/Spearman, failure
detection, task-conditioned data filtering, and five metrics-vs-step plots.
The implementation is `experiments/natural_dual_eval/sweep_natural_dual_test.py`.

Do not evaluate a checkpoint until its directory contains `config.json` and all
model shard files. The input test cache is:

```text
/data/yingxi/robometer/natural_progress_test_20260914_2gpu/robometer_test_processed/local_natural_dual_test/processed_dataset
```

For a two-GPU parallel sweep, start one worker per GPU with disjoint checkpoint
steps. Use `--steps` to assign step 0 and odd/even checkpoint groups, and write
each worker to its own output subdirectory. Example for the v3 run:

```bash
REPO=/data/yingxi/RoboFPE
PY=$REPO/robometer/.venv/bin/python
TEST=/data/yingxi/robometer/natural_progress_test_20260914_2gpu/robometer_test_processed/local_natural_dual_test/processed_dataset
CKPTS=/data/yingxi/robometer/natural_progress_collection_20260914_4gpu/robometer_train_save50_v3_logs_b4/natural_dual_fsdp_full_500_save50_v3
OUT=/data/yingxi/robometer/natural_progress_test_20260914_2gpu/robometer_checkpoint_sweep_save50_v3_parallel

CUDA_VISIBLE_DEVICES=0 $PY $REPO/experiments/natural_dual_eval/sweep_natural_dual_test.py \
  --processed-dataset "$TEST" --checkpoint-root "$CKPTS" \
  --base-model /data/yingxi/robometer/robometer-4b_basefixed \
  --output-dir "$OUT/worker0" --per-class 50 --steps 0 50 150 250 350 450 &
CUDA_VISIBLE_DEVICES=1 $PY $REPO/experiments/natural_dual_eval/sweep_natural_dual_test.py \
  --processed-dataset "$TEST" --checkpoint-root "$CKPTS" \
  --base-model /data/yingxi/robometer/robometer-4b_basefixed \
  --output-dir "$OUT/worker1" --per-class 50 --steps 100 200 300 400 500 &
wait
```

After both workers exit, merge their `metrics.json` arrays by `step`, write the
combined `metrics.json` and `metrics.csv` at `$OUT`, then invoke the sweep
script's `write_plots()` helper on the merged metrics. The required artifacts
are `metrics_vs_step_progress_mse.png`,
`metrics_vs_step_progress_spearman.png`, two failure-detection plots, and the
filtering plot. Keep each worker's `step_*_details.json` as the audit trail.

The single-machine preparation script `experiments/natural_dual_eval/prepare_and_sweep.sh`
builds the exact-render cache but contains a historical checkpoint-root default;
override it to the run being evaluated rather than using it unchanged.
