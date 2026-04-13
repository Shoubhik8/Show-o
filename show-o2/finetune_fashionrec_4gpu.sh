#!/bin/bash
# Finetune Show-o2 1.5B for MMU on 4x A40 (48GB) using DeepSpeed ZeRO-2 + bf16.
# To fall back to ZeRO-3 (param sharding), point --config_file at
#   ../accelerate_configs/4_gpus_deepspeed_zero3.yaml instead.

cd "$(dirname "$0")"
accelerate launch \
  --config_file ../accelerate_configs/4_gpus_deepspeed_zero2.yaml \
  --main_process_port=9999 \
  finetune_fashionrec.py \
  config=configs/fashionRec_1.5b_finetune.yaml
