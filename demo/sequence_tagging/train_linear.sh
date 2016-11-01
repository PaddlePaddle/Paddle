#!/bin/bash

paddle train \
       --config linear_crf.py \
       --use_gpu=0 \
       --dot_period=100 \
       --log_period=10000 \
       --test_period=0 \
       --num_passes=10
