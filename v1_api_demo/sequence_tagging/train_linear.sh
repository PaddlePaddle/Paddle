#!/bin/bash

paddle train \
       --config linear_crf.py \
       --use_gpu=0 \
       --dot_period=100 \
       --log_period=10000 \
       --test_period=0 \
       --num_passes=10
2>&1 | tee 'train_linear.log'
paddle usage -l 'train_linear.log' -e $? -n "sequence_tagging_train_linear" >/dev/null 2>&1
