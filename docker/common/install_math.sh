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

set -ex

MT_OPENCV_URL="http://oss.mthreads.com/release-ci/Math-X/mt_opencv.tar.gz"
MU_RAND_URL="http://oss.mthreads.com/release-ci/Math-X/muRAND_dev1.0.0.tar.gz"
MU_SPARSE_URL="http://oss.mthreads.com/release-ci/Math-X/muSPARSE_dev0.1.0.tar.gz"
MU_ALG_URL="http://oss.mthreads.com/release-ci/Math-X/muAlg_dev-0.1.1-Linux.deb"
MU_TRUST_URL="http://oss.mthreads.com/release-ci/Math-X/muThrust_dev-0.1.1-Linux.deb"
WORK_DIR="${PWD}"
DATE=$(date +%Y%m%d)

# parse parameters
parameters=`getopt -o h:: --long mt_opencv_url:,mu_rand_url:,mu_sparse_url:,mu_alg_url:,mu_trust_url:,help, -n "$0" -- "$@"`
[ $? -ne 0 ] && exit 1

eval set -- "$parameters"

while true;do
  case "$1" in
    --mt_opencv_url) MT_OPENCV_URL=$2; shift 2;;
    --mu_rand_url)   MU_RAND_URL=$2; shift 2;;
    --mu_sparse_url) MU_SPARSE_URL=$2; shift 2;;
    --mu_alg_url)    MU_ALG_URL=$2; shift 2;;
    --mu_trust_url) MU_TRUST_URL=$2; shift 2;;
    --) shift ; break ;;
    *) exit 1 ;;
  esac
done

install_mu_rand() {
  if [ -d $1 ]; then
    rm -rf $1/mu_rand*.tar.gz
  fi
  echo -e "\033[34mDownloading mu_rand.tar.gz to $1\033[0m"
  wget --no-check-certificate $MU_RAND_URL -O $1/mu_rand.tar.gz
  if [ -d $1/muRand ]; then
    rm -rf $1/muRand/*
  fi
  mkdir -p $1/muRand
  tar -zxf $1/mu_rand.tar.gz -C $1/muRand
  INSTALL_DIR=$(dirname $(find $1/muRand -name install.sh))
  pushd $INSTALL_DIR
  sudo bash install.sh
  popd
}

install_mu_sparse() {
  if [ -d $1 ]; then
    rm -rf $1/mu_sparse*.tar.gz
  fi
  echo -e "\033[34mDownloading mu_sparse.tar.gz to $1\033[0m"
  wget --no-check-certificate $MU_SPARSE_URL -O $1/mu_sparse.tar.gz
  if [ -d $1/muSparse ]; then
    rm -rf $1/muSparse/*
  fi
  mkdir -p $1/muSparse
  tar -zxf $1/mu_sparse.tar.gz -C $1/muSparse
  INSTALL_DIR=$(dirname $(find $1/muSparse -name install.sh))
  pushd $INSTALL_DIR
  sudo bash install.sh
  popd
}

install_mu_alg_url() {
  if [ -d $1 ]; then
    rm -rf $1/mu_alg*.deb
  fi
  echo -e "\033[34mDownloading mu_alg.deb to $1\033[0m"
  wget --no-check-certificate $MU_ALG_URL -O $1/mu_alg.deb
  sudo dpkg -i $1/mu_alg.deb
}

install_trust_url() {
  if [ -d $1 ]; then
    rm -rf $1/mu_thrust*.deb
  fi
  echo -e "\033[34mDownloading mu_thrust.deb to $1\033[0m"
  wget --no-check-certificate $MU_TRUST_URL -O $1/mu_thrust.deb
  sudo dpkg -i $1/mu_thrust.deb
}

main() {
  # Get all install function names
  function_names=$(grep "^install" $0 | sed -nE 's/^([a-zA-Z0-9_]+)\(.*/\1/p')
  mkdir -p $WORK_DIR/$DATE
  for fn_name in $function_names; do
    eval $fn_name $WORK_DIR/$DATE
  done
  pushd ~
  sudo rm -rf $WORK_DIR/$DATE
  popd
}

main
