#!/bin/bash
set -xe

REPO="${REPO:-paddlepaddle}"

sed 's/<baseimg>/10.1-cudnn7-devel-centos6/g' Dockerfile.x64.gcc8 | \
sed 's/<NCCL_MAKE_OPTS>/NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75"/g'> Dockerfile.tmp
docker build --rm=true --no-cache --disable-content-trust=true -t ${REPO}/paddle_manylinux_devel:cuda10.1_cudnn7 -f Dockerfile.tmp .
# docker push ${REPO}/paddle_manylinux_devel:cuda10.0_cudnn7
