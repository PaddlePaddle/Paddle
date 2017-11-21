#!/bin/bash

python paddle.py \
  --job_dispatch_package="/root/wuyi/fabric_submit/workspace" \
  --dot_period=10 \
  --ports_num_for_sparse=1 \
  --log_period=50 \
  --num_passes=5 \
  --trainer_count=2 \
  --saving_period=1 \
  --local=0 \
  --config=./trainer_config.py \
  --save_dir=./output \
  --use_gpu=0
