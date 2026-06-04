# Commends
1. Split training data and evaluation data:
```
cd /root/RoboFPE/robometer
uv run python scripts/split_local_hf_dataset.py \
  --input /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset \
  --train-output /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_train \
  --eval-output /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_eval \
  --seed 42
```
Check data before training and evaluation:
```
uv run python /root/RoboFPE/mani_envs/data_collection/visualize_random_progress_samples.py \
  --dataset /data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_train \
  --num-samples 20 \
  --seed 0 \
  --output-dir ../mani_envs/data_collection/plots_train
```
2. Data preprocessing:
```
export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com

# Process train split
uv run python -m robometer.data.scripts.preprocess_local_hf_datasets  --config_path robometer/configs/preprocess_local_hf.yaml  --dataset_roots '["/data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_train/"]'  --dataset_names '["local/PegInsertionVertical_train"]'

# Process eval split
uv run python -m robometer.data.scripts.preprocess_local_hf_datasets  --config_path robometer/configs/preprocess_local_hf.yaml  --dataset_roots '["/data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset_eval/"]'  --dataset_names '["local/PegInsertionVertical_eval"]'
```
Check processed data:
```
uv run python /root/RoboFPE/mani_envs/data_collection/visualize_processed_progress_samples.py \
  --dataset /data/yingxi/robometer/local_PegInsertionVertical_train \
  --num-samples 20 \
  --seed 0 \
  --output-dir ../mani_envs/data_collection/plots_cache_train
```
3. Training: 
```
cd robometer

export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True         uv run accelerate launch           --config_file robometer/configs/distributed/fsdp.yaml           --num_processes=4           train.py           data.train_datasets=[local/PegInsertionVertical_train]           data.eval_datasets=[local/PegInsertionVertical_eval]           training.per_device_train_batch_size=16           training.per_device_eval_batch_size=16           data.max_frames=8           data.resized_height=224           data.resized_width=224           training.save_strategy=steps           training.save_steps=50           training.save_total_limit=5           training.logging_steps=1           logging.log_to=[wandb]
```
4. Evaluation
```
uv run python robometer/evals/run_baseline_eval_local_peg_insertion_vertical.py \
    reward_model=rbm \
    model_path=robometer/Robometer-4B \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=8 \
    model_config.batch_size=16
```
Compare oracle with baseline:
```
export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
export HF_ENDPOINT=https://hf-mirror.com
uv run python robometer/evals/run_baseline_eval_local_peg_insertion_vertical.py \
    reward_model=rbm \
    model_path=/data/yingxi/robometer/logs/rbm/checkpoint-800/ \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=10\
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=8 \
    model_config.batch_size=16
```
Evaluate with real-world data:
```
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
