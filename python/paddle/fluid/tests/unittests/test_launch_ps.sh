#!/bin/bash
set -e
# use default values
launch_py=${PADDLE_BINARY_DIR}/python/paddle/distributed/launch_ps.py
python ${launch_py} fleet_ps_training.py 2> ut.elog

if grep -q "server are killed" ut.elog; then
    echo "succeed"
else
    echo "failed"
    exit -1
fi
