# Commends
1. Training: 
```
cd robometer
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=8 \
  train.py \
  data.train_datasets=[local/PegInsertionVertical-v1_hf_dataset] \
  data.eval_datasets=[local/PegInsertionVertical-v1_hf_dataset] \
  training.per_device_train_batch_size=1 \
  training.per_device_eval_batch_size=1 \
  training.gradient_accumulation_steps=16 \
  data.max_frames=8 \
  data.resized_height=224 \
  data.resized_width=224 \
  logging.log_to=[]
```
