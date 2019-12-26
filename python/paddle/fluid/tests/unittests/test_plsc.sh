#!/bin/bash
set -e

# install plsc
python -m pip install plsc --force

# test
python -m paddle.distributed.launch \
        --selected_gpus="0,1" \
        --log_dir="/tmp/plsc.log" do_plsc.py
