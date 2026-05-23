# Commends
1. Training: 
```
cd robometer
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
  --config_file robometer/configs/distributed/fsdp.yaml \
  --num_processes=1 \
  train.py \
  data.train_datasets=[local/PegInsertionVertical-v1_hf_dataset] \
  data.eval_datasets=[local/PegInsertionVertical-v1_hf_dataset] 
  data.labeled_progress_data_sources=[gen_progress_success,gen_progress_failure] \
  data.max_frames=8 \
  model.train_progress_head=true \
  model.train_preference_head=true \
  model.train_success_head=true \
  training.predict_pref_progress=true \
  training.max_steps=15000
```
