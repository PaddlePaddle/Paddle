#!/bin/bash
set -e

function test() {
  cfg=$1
  batch=$2
  prefix=$3
  batch_per_gpu=`expr ${batch} / 4`
  sed -i "/input: \"data\"/{n;s/^input_dim.*/input_dim: ${batch_per_gpu}/g}" $cfg 
  sed -i "/input: \"label\"/{n;s/^input_dim.*/input_dim: ${batch_per_gpu}/g}" $cfg 
  sed -i "1c\net : \"${cfg}\"" solver.prototxt
  caffe train --solver=solver.prototxt -gpu 0,1,2,3 > logs/${prefix}-4gpu-batch${batch}.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

# alexnet
test alexnet.prototxt 512 alexnet 
test alexnet.prototxt 1024 alexnet 

# googlnet 
test googlenet.prototxt 512 googlenet 
