#!/bin/bash
# This script benchmarking the PaddlePaddle Fluid on
# single thread single GPU.

mkdir -p logs
#export FLAGS_fraction_of_gpu_memory_to_use=0.0
export CUDNN_PATH=/paddle/cudnn_v5

# disable openmp and mkl parallel
#https://github.com/PaddlePaddle/Paddle/issues/7199
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
ht=`lscpu |grep "per core"|awk -F':' '{print $2}'|xargs`
if [ $ht -eq 1 ]; then # HT is OFF
    if [ -z "$KMP_AFFINITY" ]; then
        export KMP_AFFINITY="granularity=fine,compact,0,0"
    fi
    if [ -z "$OMP_DYNAMIC" ]; then
        export OMP_DYNAMIC="FALSE"
    fi
else # HT is ON
    if [ -z "$KMP_AFFINITY" ]; then
        export KMP_AFFINITY="granularity=fine,compact,1,0"
    fi
fi
# disable multi-gpu if have more than one
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH

# only query the gpu used
nohup stdbuf -oL nvidia-smi \
      --id=${CUDA_VISIBLE_DEVICES} \
      --query-gpu=timestamp \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv \
      --filename=mem.log  \
      -l 1 &

# mnist
# mnist gpu mnist 128
FLAGS_benchmark=true stdbuf -oL python fluid_benchmark.py \
               --model=mnist \
               --device=GPU \
               --batch_size=128 \
               --skip_batch_num=5 \
               --iterations=500 \
               2>&1 | tee -a logs/mnist_gpu_128.log

# vgg16
# gpu cifar10 128
FLAGS_benchmark=true stdbuf -oL python fluid_benchmark.py \
               --model=vgg16 \
               --device=GPU \
               --batch_size=128 \
               --skip_batch_num=5 \
               --iterations=30 \
               2>&1 | tee -a logs/vgg16_gpu_128.log

# flowers gpu  128
FLAGS_benchmark=true stdbuf -oL python fluid_benchmark.py \
               --model=vgg16 \
               --device=GPU \
               --batch_size=32 \
               --data_set=flowers \
               --skip_batch_num=5 \
               --iterations=30 \
               2>&1 | tee -a logs/vgg16_gpu_flowers_32.log

# resnet50
# resnet50 gpu cifar10 128
FLAGS_benchmark=true stdbuf -oL python fluid_benchmark.py \
               --model=resnet \
               --device=GPU \
               --batch_size=128 \
               --data_set=cifar10 \
               --skip_batch_num=5 \
               --iterations=30 \
               2>&1 | tee -a logs/resnet50_gpu_128.log

# resnet50 gpu flowers 64
FLAGS_benchmark=true stdbuf -oL python fluid_benchmark.py \
               --model=resnet \
               --device=GPU \
               --batch_size=64 \
               --data_set=flowers \
               --skip_batch_num=5 \
               --iterations=30 \
               2>&1 | tee -a logs/resnet50_gpu_flowers_64.log

# lstm
# lstm gpu imdb 32 # tensorflow only support batch=32
FLAGS_benchmark=true stdbuf -oL python fluid_benchmark.py \
               --model=stacked_dynamic_lstm \
               --device=GPU \
               --batch_size=32 \
               --skip_batch_num=5 \
               --iterations=30 \
               2>&1 | tee -a logs/lstm_gpu_32.log

# seq2seq
# seq2seq gpu wmb 128
FLAGS_benchmark=true stdbuf -oL python fluid_benchmark.py \
               --model=machine_translation \
               --device=GPU \
               --batch_size=128 \
               --skip_batch_num=5 \
               --iterations=30 \
               2>&1 | tee -a logs/lstm_gpu_128.log
