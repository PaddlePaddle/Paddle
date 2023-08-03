#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mccl will be integrated into musa_toolkits soon
set -ex

#MCCL_URL="http://oss.mthreads.com/release-rc/cuda_compatible/rc1.3.0/mccl_rc1.1.0.txz"
MCCL_URL="https://oss.mthreads.com/release-ci/computeQA/tmp/20230627_MCCL_tmp_pkg/MCCL/mccl.tar.gz"
WORK_DIR="${PWD}"
DATE=$(date +%Y%m%d)

# parse parameters
parameters=`getopt -o h:: --long mccl_url:,help, -n "$0" -- "$@"`
[ $? -ne 0 ] && exit 1

eval set -- "$parameters"

while true;do
  case "$1" in
    --mccl_url) MCCL_URL=$2; shift 2;;
    --) shift ; break ;;
    *) exit 1 ;;
  esac
done

install_mccl() {
  if [ -d $1 ]; then
    rm -rf $1/mccl*.tar.gz
  fi
  echo -e "\033[34mDownloading mccl.txz to $1\033[0m"
  wget --no-check-certificate $MCCL_URL -O $1/mccl.tar.gz
  if [ -d $1/mccl ]; then
    rm -rf $1/mccl/*
  fi
  mkdir -p $1/mccl
  tar zxvf $1/mccl.tar.gz -C $1/mccl
  INSTALL_DIR=$(dirname $(find $1/mccl -name install.sh))
  pushd $INSTALL_DIR
  sudo bash install.sh
  popd
}


mkdir -p $WORK_DIR/$DATE
install_mccl $WORK_DIR/$DATE
pushd ~
sudo rm -rf $WORK_DIR/$DATE
popd
