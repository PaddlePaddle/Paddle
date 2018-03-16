#!/bin/bash
#
# This build script is used for TeamCity CI, we need to 
# set the follwoing parameters in the project configuration:
#   PARAMETER            | EXAMPLE    |   COMMENTS
# CTEST_OUTPUT_ON_FAILURE   1         // whether output failure message
# CTEST_PARALLEL_LEVEL      5         // parallel level for running ctest
# PADDLE_VERSION            0.11.0    // the latest tag
# WITH_AVX                  ON        // wheter the instruction support AVX
# WITH_GOLANG               OFF       // don't build the Golang code
# WITH_GPU                  ON        // running on GPU
# WITH_UBUNTU_MIRROR        ON        // use unbuntu mirro to speed up dev Docker image building
#

set -xe

PR_ID=$(echo %teamcity.build.vcs.branch.Paddle_PaddlePaddle% | sed 's/[^0-9]*//g')
echo ${PR_ID}

CHANGED_FILES=$(cat %system.teamcity.build.changedFiles.file% | awk -F : '{print $1}')
if [ -z "$CHANGED_FILES" ]
then
  echo "empty changed files"
else
  echo "changed files: $CHANGED_FILES"
  COUNT=$(echo $CHANGED_FILES | awk -F . '{print $NF}'|grep -i -E 'md|rst|jpg|png' | wc -l)
  FILE_COUNT=$(echo $CHANGED_FILES | wc -l)
  if [ "$COUNT" -eq "$FILE_COUNT" ]
  then
    echo "all readme files or pictures, skip CI"
    exit 0
  fi
fi

apt_mirror='s#http://archive.ubuntu.com/ubuntu#mirror://mirrors.ubuntu.com/mirrors.txt#g'

export PADDLE_DEV_NAME=paddlepaddle/paddle:dev

nvidia-docker run -i --rm -v $PWD:/paddle ${PADDLE_DEV_NAME} rm -rf /paddle/build

MD5_EXT_CMAKE=$(cat ./cmake/external/*.cmake |md5sum | awk '{print $1}')

flag_update_tp_cache=false
if [ ! -d /root/.cache/third_party ];then
    mkdir -p /root/.cache/third_party
fi
if [ -f /root/.cache/third_party/${MD5_EXT_CMAKE}.tar.gz ];then
    mkdir -p ${PWD}/build
    tar zxvf /root/.cache/third_party/${MD5_EXT_CMAKE}.tar.gz -C $PWD
else
    # clear older tar files if MD5 has chanaged.
    rm -rf /root/.cache/third_party/*.tar.gz
    flag_update_tp_cache=true
fi

# Do not build dev image for now

if [[ "%WITH_UBUNTU_MIRROR%" == "ON" ]]; then
   docker build -t ${PADDLE_DEV_NAME} --build-arg UBUNTU_MIRROR=mirror://mirrors.ubuntu.com/mirrors.txt .
else
   docker build -t ${PADDLE_DEV_NAME} .
fi

nvidia-docker run -i --rm -v $PWD:/paddle ${PADDLE_DEV_NAME} rm -rf /paddle/third_party /paddle/build

# make sure CPU only build with openblas passes.
nvidia-docker run -i --rm -v $PWD:/paddle -v /root/.cache:/root/.cache\
    -e "APT_MIRROR=${apt_mirror}"\
    -e "WITH_GPU=OFF"\
    -e "WITH_AVX=%WITH_AVX%"\
    -e "WITH_GOLANG=%WITH_GOLANG%"\
    -e "WITH_TESTING=OFF"\
    -e "WITH_COVERAGE=OFF" \
    -e "COVERALLS_UPLOAD=OFF" \
    -e "GIT_PR_ID=${PR_ID}" \
    -e "WITH_C_API=%WITH_C_API%" \
    -e "CMAKE_BUILD_TYPE=Debug" \
    -e "WITH_MKL=OFF" \
    -e "JSON_REPO_TOKEN=JSUOs6TF6fD2i30OJ5o2S55V8XWv6euen" \
    -e "WITH_DEB=OFF"\
    -e "PADDLE_VERSION=%PADDLE_VERSION%"\
    -e "PADDLE_FRACTION_GPU_MEMORY_TO_USE=0.15" \
    -e "RUN_TEST=OFF" ${PADDLE_DEV_NAME}

# run build with GPU and intel MKL and test.

nvidia-docker run -i --rm -v $PWD:/paddle -v /root/.cache:/root/.cache\
    -e "FLAGS_fraction_of_gpu_memory_to_use=0.15"\
    -e "CTEST_OUTPUT_ON_FAILURE=%CTEST_OUTPUT_ON_FAILURE%"\
    -e "CTEST_PARALLEL_LEVEL=%CTEST_PARALLEL_LEVEL%"\
    -e "APT_MIRROR=${apt_mirror}"\
    -e "WITH_GPU=%WITH_GPU%"\
    -e "CUDA_ARCH_NAME=Auto"\
    -e "WITH_AVX=%WITH_AVX%"\
    -e "WITH_GOLANG=%WITH_GOLANG%"\
    -e "WITH_TESTING=ON"\
    -e "WITH_C_API=%WITH_C_API%" \
    -e "WITH_COVERAGE=ON" \
    -e "COVERALLS_UPLOAD=ON" \
    -e "GIT_PR_ID=${PR_ID}" \
    -e "JSON_REPO_TOKEN=JSUOs6TF6fD2i30OJ5o2S55V8XWv6euen" \
    -e "WITH_DEB=OFF"\
    -e "PADDLE_VERSION=%PADDLE_VERSION%"\
    -e "PADDLE_FRACTION_GPU_MEMORY_TO_USE=0.15" \
    -e "CUDA_VISIBLE_DEVICES=0,1" \
    -e "RUN_TEST=ON" ${PADDLE_DEV_NAME}


if $flag_update_tp_cache; then
    tar czvf /root/.cache/third_party/${MD5_EXT_CMAKE}.tar.gz ./build/third_party
fi
