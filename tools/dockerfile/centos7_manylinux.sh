#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

set -xe

REPO="${REPO:-paddledocker}"

function make_cuda9cudnn7(){
  sed 's/<baseimg>/9.0-cudnn7-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
}


function make_cuda10cudnn7() {
  sed 's/<baseimg>/10.0-cudnn7-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
}


function make_cuda101cudnn7() {
  sed 's/<baseimg>/10.1-cudnn7-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc82 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-8.2/bin:\$PATH \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp 
}

function make_cuda102cudnn7() {
  sed 's/<baseimg>/10.2-cudnn7-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc82 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-8.2/bin:\$PATH \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
}

function make_cuda102cudnn8() {
  sed 's/<baseimg>/10.2-cudnn8-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc82 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-8.2/bin:\$PATH \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
}

function make_cuda11cudnn8() {
  sed 's/<baseimg>/11.0-cudnn8-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc82 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-8.2/bin:\$PATH \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
}

function main() {
  local CMD=$1 
  case $CMD in
    cuda9cudnn7)
      make_cuda9cudnn7
      ;;
    cuda10cudnn7)
      make_cuda10cudnn7
      ;;
    cuda101cudnn7)
      make_cuda101cudnn7
      ;;
    cuda102cudnn7)
      make_cuda102cudnn7
      ;;
    cuda102cudnn8)
      make_cuda102cudnn8
      ;;
    cuda11cudnn8)
      make_cuda11cudnn8
     ;;
    *)
      echo "Make dockerfile error, Without this paramet."
      exit 1
      ;;
  esac
}

main "$@"
