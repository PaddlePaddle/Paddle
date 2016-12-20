#!/bin/sh

PATH_TO_LOCAL_WORKSPACE=/home/sparse_test/workspace


mv cluster_config.conf conf.py

python paddle.py \
  --job_dispatch_package="${PATH_TO_LOCAL_WORKSPACE}" \
  --use_gpu=0 \
  --config=./sparse_trainer_config.py \
  --saving_period=1 \
  --test_period=0 \
  --num_passes=4 \
  --dot_period=2 \
  --log_period=20 \
  --trainer_count=10 \
  --saving_period_by_batches=5000 \
  --ports_num_for_sparse=4 \
  --local=0 \
