#!/bin/bash
set -e

function test() {
  cfg=$1
  batch=$2
  prefix=$3
  batch_per_gpu=`expr ${batch} / 4`
  echo "batch size per gpu is ${batch_per_gpu}"
  sed -i "/input: \"data\"/{n;s/^input_dim.*/input_dim: ${batch_per_gpu}/g}" $cfg 
  sed -i "/input: \"label\"/{n;s/^input_dim.*/input_dim: ${batch_per_gpu}/g}" $cfg 
  caffe time --model=$cfg --iterations=50 --gpu=0,1,2,3 --logtostderr=1 > logs/${prefix}-4gpu-batch${batch}.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

# alexnet
test alexnet.prototxt 128 alexnet 
test alexnet.prototxt 256 alexnet 
test alexnet.prototxt 512 alexnet 
