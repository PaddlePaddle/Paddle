#!/bin/bash
set -xe

BaseImg=10.0-cudnn7-devel-centos7
GCC=gcc48
PY_ABI=cp35-cp35m
PY_VER=3.5

REPO="${REPO:-paddledocker}"

sed 's/<baseimg>/${BaseImg}/g' Dockerfile.centos.${GCC} | \
sed 's/<NCCL_MAKE_OPTS>/NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62 -gencode=arch=compute_70,code=sm_70"/g'> Dockerfile.tmp
docker build -t ${REPO}/paddle_manylinux_devel:test -f Dockerfile.tmp .

docker run -i --rm -v $PWD:/paddle ${REPO}/paddle_manylinux_devel:test \
  rm -rf /paddle/third_party /paddle/build
export GIT_PATH="%system.agent.home.dir%"/system/git

docker run -i --rm -v $PWD:/paddle \
  -v ${GIT_PATH}:${GIT_PATH}\
  -w /paddle \
  -e "CMAKE_BUILD_TYPE=Release" \
  -e "PYTHON_ABI=${PY_ABI}" \
  -e "PADDLE_VERSION=0.0.0" \
  -e "WITH_DOC=OFF" \
  -e "WITH_AVX=ON" \
  -e "WITH_GPU=ON" \
  -e "WITH_TEST=OFF" \
  -e "RUN_TEST=OFF" \
  -e "WITH_SWIG_PY=ON" \
  -e "WITH_PYTHON=ON" \
  -e "WITH_C_API=OFF" \
  -e "WITH_STYLE_CHECK=OFF" \
  -e "WITH_TESTING=OFF" \
  -e "CMAKE_EXPORT_COMPILE_COMMANDS=ON" \
  -e "WITH_MKL=ON" \
  -e "BUILD_TYPE=Release" \
  -e "WITH_DISTRIBUTE=ON" \
  -e "WITH_FLUID_ONLY=ON" \
  -e "PY_VERSION=${PY_VER}" \
  ${REPO}/paddle_manylinux_devel:test \
  /bin/bash -c "paddle/scripts/paddle_build.sh combine_avx_noavx"
cp ./build/python/dist/paddlepaddle*.whl ./output

docker image rm ${REPO}/paddle_manylinux_devel:test
