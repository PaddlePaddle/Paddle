#!/bin/bash
set -e
# use default values
python -m paddle.distributed.launch_ps fleet_ps_training.py 2> ut.elog

if grep -q "server are killed" ut.elog; then
    echo "succeed"
else
    echo "failed"
    exit -1
fi
