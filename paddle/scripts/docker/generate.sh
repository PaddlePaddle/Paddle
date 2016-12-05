#!/bin/bash

set -e
cd `dirname $0`

m4 -DPADDLE_WITH_GPU=OFF \
   -DPADDLE_WITH_AVX=ON  \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 \
   Dockerfile.m4 > Dockerfile.cpu

m4 -DPADDLE_WITH_GPU=OFF \
   -DPADDLE_WITH_AVX=OFF \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 \
   Dockerfile.m4 > Dockerfile.cpu-noavx

m4 -DPADDLE_WITH_GPU=ON \
   -DPADDLE_WITH_AVX=ON \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   Dockerfile.m4 > Dockerfile.gpu

m4 -DPADDLE_WITH_GPU=ON  \
   -DPADDLE_WITH_AVX=OFF \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   Dockerfile.m4 > Dockerfile.gpu-noavx
