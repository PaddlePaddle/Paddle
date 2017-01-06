#!/bin/bash

paddle train \
       --config rnn_crf.py \
       --parallel_nn=1 \
       --use_gpu=1 \
       --dot_period=10 \
       --log_period=1000 \
       --test_period=0 \
       --num_passes=10 \
2>&1 | tee 'train.log'
paddle usage -l 'train.log' -e $? -n "sequence_tagging_train" >/dev/null 2>&1
