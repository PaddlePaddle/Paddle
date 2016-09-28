#!/bin/bash
set -e
cd `dirname $0`
m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=OFF -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=ON\
   Dockerfile.m4 > Dockerfile.cpu

m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=OFF -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=OFF\
   Dockerfile.m4 > Dockerfile.cpu-noavx

m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=OFF\
   Dockerfile.m4 > Dockerfile.cpu-noavx-devel

m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=ON\
   Dockerfile.m4 > Dockerfile.cpu-devel


m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=ON \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=ON\
   Dockerfile.m4 > Dockerfile.cpu-demo

m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=ON \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=OFF\
   Dockerfile.m4 > Dockerfile.cpu-noavx-demo


m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=OFF -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=ON \
   Dockerfile.m4 > Dockerfile.gpu

m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=OFF -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=OFF \
   Dockerfile.m4 > Dockerfile.gpu-noavx


m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=ON \
   Dockerfile.m4 > Dockerfile.gpu-devel

m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=OFF \
   Dockerfile.m4 > Dockerfile.gpu-noavx-devel

m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=ON \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=ON \
   Dockerfile.m4 > Dockerfile.gpu-demo


m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=ON \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=OFF \
   Dockerfile.m4 > Dockerfile.gpu-noavx-demo

