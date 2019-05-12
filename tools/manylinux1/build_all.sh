#!/bin/bash
set -xe

REPO="${REPO:-typhoon1986}"

# NOTE: version matches are determined!
sed 's/<baseimg>/7.5-cudnn5-devel-centos6/g' Dockerfile.x64 | \
sed 's/<NCCL_MAKE_OPTS>/NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52"/g'> Dockerfile.tmp
docker build -t ${REPO}/paddle_manylinux_devel:cuda7.5_cudnn5 -f Dockerfile.tmp .
docker push ${REPO}/paddle_manylinux_devel:cuda7.5_cudnn5

sed 's/<baseimg>/8.0-cudnn5-devel-centos6/g' Dockerfile.x64 | \
sed 's/<NCCL_MAKE_OPTS>/NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62"/g'> Dockerfile.tmp
docker build -t ${REPO}/paddle_manylinux_devel:cuda8.0_cudnn5 -f Dockerfile.tmp .
docker push ${REPO}/paddle_manylinux_devel:cuda8.0_cudnn5

sed 's/<baseimg>/8.0-cudnn7-devel-centos6/g' Dockerfile.x64 | \
sed 's/<NCCL_MAKE_OPTS>/NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62"/g'> Dockerfile.tmp

docker build -t ${REPO}/paddle_manylinux_devel:cuda8.0_cudnn7 -f Dockerfile.tmp .
docker push ${REPO}/paddle_manylinux_devel:cuda8.0_cudnn7

sed 's/<baseimg>/9.0-cudnn7-devel-centos6/g' Dockerfile.x64 | \
sed 's/<NCCL_MAKE_OPTS>/NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62 -gencode=arch=compute_70,code=sm_70"/g'> Dockerfile.tmp
docker build -t ${REPO}/paddle_manylinux_devel:cuda9.0_cudnn7 -f Dockerfile.tmp .
docker push ${REPO}/paddle_manylinux_devel:cuda9.0_cudnn7

sed 's/<baseimg>/10.0-devel-centos6/g' Dockerfile.x64 | \
sed 's/<NCCL_MAKE_OPTS>/NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75"/g'> Dockerfile.tmp
docker build -t ${REPO}/paddle_manylinux_devel:cuda10.0_cudnn7 -f Dockerfile.tmp .
docker push ${REPO}/paddle_manylinux_devel:cuda10.0_cudnn7
