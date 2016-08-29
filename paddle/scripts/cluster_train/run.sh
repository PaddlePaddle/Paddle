#!/bin/sh

#python paddle.py \
#  --job_workspace="${PATH_TO_REMOTE_EXISTED_WORKSPACE}" \
#  --dot_period=10 \
#  --ports_num_for_sparse=2 \
#  --log_period=50 \
#  --num_passes=10 \
#  --trainer_count=4 \
#  --saving_period=1 \
#  --local=0 \
#  --config=./trainer_config.py \
#  --save_dir=./output \
#  --use_gpu=0

python paddle.py \
  --job_dispatch_package="${PATH_TO_LOCAL_WORKSPACE}" \
  --dot_period=10 \
  --ports_num_for_sparse=2 \
  --log_period=50 \
  --num_passes=10 \
  --trainer_count=4 \
  --saving_period=1 \
  --local=0 \
  --config=./trainer_config.py \
  --save_dir=./output \
  --use_gpu=0
