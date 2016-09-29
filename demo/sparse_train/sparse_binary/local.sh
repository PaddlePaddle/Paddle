#!/bin/sh 

set -x

paddle train \
    --use_gpu=0 \
    --config=./sparse_trainer_config.py \
    --saving_period=1 \
    --test_period=0 \
    --num_passes=4 \
    --dot_period=2 \
    --log_period=20 \
    --trainer_count=10 \
    --saving_period_by_batches=5000 \
    --local=1
