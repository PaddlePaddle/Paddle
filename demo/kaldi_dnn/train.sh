#!/usr/bin/env bash
set -e

cfg=/paddle/demo/kaldi_dnn/trainer_config.dnn.py

sudo docker run -d \
  -v /work/local/lipeng/github/unisound-ail/paddle/demo:/paddle/demo \
  -v /work/local/paddle/data:/paddle/data \
  paddledev/paddle:cpu-latest paddle train \
  --config=$cfg \
  --save_dir=/paddle/demo/kaldi_dnn/output \
  --trainer_count=4 \
  --log_period=20 \
  --num_passes=15 \
  --use_gpu=false \
  --show_parameter_stats_period=100 \
  --test_all_data_in_one_period=1 \
  2>&1 | tee 'train.log'

