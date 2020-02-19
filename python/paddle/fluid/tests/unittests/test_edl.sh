#!/bin/bash
set -e

unset http_proxy https_proxy

# running under edl
export PADDLE_RUNING_ENV=PADDLE_EDL
export PADDLE_JOBSERVER="127.0.0.1:6070"
export PADDLE_JOB_ID="test_job_id_1234"
export PADDLE_POD_ID=0

nohup python job_server_demo.py &

CUDA_VISIBLE_DEVICES=0 python -m paddle.distributed.launch --log_level 10 edl_demo.py > edl_demo.log 2>&1

echo "test request and response"
str=""
file=edl_demo.log

if grep -q "$str" "$file"; then
    echo "request and response ok!"
else
    echo "request and response error!"
    exit -1
fi

exit 0

# kill job_server_demo process
curl -i -X POST -H "'Content-type':'application/json'" -d '{}' $PADDLE_JOBSERVER
