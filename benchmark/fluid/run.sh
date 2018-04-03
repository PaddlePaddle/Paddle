#!/bin/bash
# This script benchmarking the PaddlePaddle Fluid on
# single thread single GPU.
export CUDNN_PATH=/paddle/cudnn_v5/cuda/lib

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


# vgg16
# cifar10 gpu cifar10 128
FLAGS_benchmark=true python fluid/vgg.py \
               --device=GPU \
               --batch_size=128 \
               --skip_batch_num=5 \
               --iterations=30  \
               2>&1 > vgg16_gpu_128.log

# resnet50
# resnet50 gpu cifar10 128
FLAGS_benchmark=true python fluid/resnet.py \
               --device=GPU \
               --batch_size=128 \
               --data_set=cifar10 \
               --model=resnet_cifar10 \
               --skip_batch_num=5 \
               --iterations=30 \
               2>&1 > resnet50_gpu_128.log

# lstm
