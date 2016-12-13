#!/bin/sh
set -eu

jobconfig=${JOB_PATH}"/"${JOB_NAME}"/"${TRAIN_CONFIG_DIR}
cd /root
cp -rf $jobconfig .
cd $TRAIN_CONFIG_DIR


python /root/start_paddle.py \
  --dot_period=10 \
  --ports_num_for_sparse=$CONF_PADDLE_PORTS_NUM \
  --log_period=50 \
  --num_passes=10 \
  --trainer_count=4 \
  --saving_period=1 \
  --local=0 \
  --config=./trainer_config.py \
  --use_gpu=0
