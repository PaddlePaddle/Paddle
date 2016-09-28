#!/bin/bash
set -e
cd `dirname $0`
m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=OFF -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=ON\
   Dockerfile.m4 > cpu/Dockerfile
cp build.sh cpu/

m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=OFF -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=OFF\
   Dockerfile.m4 > cpu-noavx/Dockerfile
cp build.sh cpu-noavx/

m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=OFF\
   Dockerfile.m4 > cpu-noavx-devel/Dockerfile
cp build.sh cpu-noavx-devel/

m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=ON\
   Dockerfile.m4 > cpu-devel/Dockerfile
cp build.sh cpu-devel/


m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=ON \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=ON\
   Dockerfile.m4 > cpu-demo/Dockerfile
cp build.sh cpu-demo/

m4 -DPADDLE_WITH_GPU=OFF -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=ON \
   -DPADDLE_BASE_IMAGE=ubuntu:14.04 -DPADDLE_WITH_AVX=OFF\
   Dockerfile.m4 > cpu-noavx-demo/Dockerfile
cp build.sh cpu-noavx-demo/


m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=OFF -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=ON \
   Dockerfile.m4 > gpu/Dockerfile
cp build.sh gpu/

m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=OFF -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=OFF \
   Dockerfile.m4 > gpu-noavx/Dockerfile
cp build.sh gpu-noavx/


m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=ON \
   Dockerfile.m4 > gpu-devel/Dockerfile
cp build.sh gpu-devel/

m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=OFF \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=OFF \
   Dockerfile.m4 > gpu-noavx-devel/Dockerfile
cp build.sh gpu-noavx-devel/

m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=ON \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=ON \
   Dockerfile.m4 > gpu-demo/Dockerfile
cp build.sh gpu-demo/


m4 -DPADDLE_WITH_GPU=ON -DPADDLE_IS_DEVEL=ON -DPADDLE_WITH_DEMO=ON \
   -DPADDLE_BASE_IMAGE=nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 \
   -DPADDLE_WITH_AVX=OFF \
   Dockerfile.m4 > gpu-noavx-demo/Dockerfile
cp build.sh gpu-noavx-demo/


