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
2. Data preprocessing:
```
uv run python -m robometer.data.scripts.preprocess_local_hf_datasets --config robometer/configs/preprocess_local_hf.yaml
export ROBOMETER_PROCESSED_DATASETS_PATH=/data/yingxi/robometer
```
3. Training: 
```
cd robometer
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True         uv run accelerate launch           --config_file robometer/configs/distributed/fsdp.yaml           --num_processes=4           train.py           data.train_datasets=[local/PegInsertionVertical-v1_hf_dataset]           data.eval_datasets=[local/PegInsertionVertical-v1_hf_dataset]           training.per_device_train_batch_size=1           training.per_device_eval_batch_size=1           training.gradient_accumulation_steps=16           data.max_frames=8           data.resized_height=224           data.resized_width=224           training.save_strategy=steps           training.save_steps=50           training.save_total_limit=1           training.logging_steps=1           logging.log_to=[wandb]
```
