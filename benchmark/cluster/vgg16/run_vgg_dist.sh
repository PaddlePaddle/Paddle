#!/bin/bash

# Update to point to the source file.
VGG_SRC="vgg16_fluid.py"

export TRAINING_ROLE=PSERVER
export TRAINERS=2
export POD_IP=127.0.0.1
export PADDLE_INIT_PORT=6174
MKL_NUM_THREADS=1 python -u ${VGG_SRC} --local 0 --ps_host=127.0.0.1:6174 --trainer_hosts=127.0.0.1:6174 &

# Need to wait for the ps to start first.
sleep 10
echo "done start ps"

export TRAINING_ROLE=TRAINER
export TRAINERS=2
export POD_IP=127.0.0.1
export PADDLE_INIT_PORT=6174
CUDA_VISIBLE_DEVICES=4 MKL_NUM_THREADS=1 python -u ${VGG_SRC} --local 0 --ps_host=127.0.0.1:6174 --trainer_hosts=127.0.0.1:6174 --device=GPU --task_index=0 &
CUDA_VISIBLE_DEVICES=5 MKL_NUM_THREADS=1 python -u ${VGG_SRC} --local 0 --ps_host=127.0.0.1:6174 --trainer_hosts=127.0.0.1:6174 --device=GPU --task_index=1 &
