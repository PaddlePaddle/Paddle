#!/bin/bash
# General trainning configurations

NICS=eth0
PADDLE_INIT_PORT=7164
PADDLE_INIT_PORTS_NUM=1
PADDLE_INIT_PORTS_NUM_FOR_SPARSE=1
PADDLE_INIT_PSERVERS=$(cat machines | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/,/g')
PADDLE_INIT_USE_GPU=False

PADDLE_INIT_NUM_GRADIENT_SERVERS=${OMPI_COMM_WORLD_SIZE}
PADDLE_INIT_TRAINER_ID=${OMPI_COMM_WORLD_RANK}
PADDLE_CLUSTER_TRAIN=True

env

# start pserver
stdbuf -oL nohup paddle pserver \
  --port=$PADDLE_INIT_PORT \
  --ports_num=$PADDLE_INIT_PORTS_NUM \
  --ports_num_for_sparse=$PADDLE_INIT_PORTS_NUM_FOR_SPARSE \
  --nics=$NICS \
  --comment=paddle_cluster_pserver \
  --num_gradient_servers=$PADDLE_INIT_NUM_GRADIENT_SERVERS \
  &> logs/pserver.log &

# start trainer
# NOTE: train.py will use the above environment variables as configuration
python train.py &> logs/train.log

# kill background pservers when train finishes
ps -ef | grep pserver | awk '{print $2}' | xargs kill
