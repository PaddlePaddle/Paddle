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

VERSION=$(nvcc --version | grep release | grep -oEi "release ([0-9]+)\.([0-9])"| sed "s/release //")
if [ "$VERSION" == "10.0" ]; then
  DEB="nccl-repo-ubuntu1604-2.4.7-ga-cuda10.0_1-1_amd64.deb"
elif [ "$VERSION" == "10.2" ] || [ "$VERSION" == "10.1" ] || [ "$VERSION" == "11.0" ]; then
  if [ -f "/etc/redhat-release" ];then
    rm -f /usr/local/lib/libnccl.so 
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-2.7.8-1+cuda10.2.x86_64.rpm
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-devel-2.7.8-1+cuda10.2.x86_64.rpm
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-static-2.7.8-1+cuda10.2.x86_64.rpm
    rpm -ivh libnccl-2.7.8-1+cuda10.2.x86_64.rpm
    rpm -ivh libnccl-devel-2.7.8-1+cuda10.2.x86_64.rpm
    rpm -ivh libnccl-static-2.7.8-1+cuda10.2.x86_64.rpm && rm -f libnccl-*
    exit 0
  fi
  DEB="nccl-repo-ubuntu1604-2.7.8-ga-cuda10.2_1-1_amd64.deb"
elif [ "$VERSION" == "9.0" ]; then
  DEB="nccl-repo-ubuntu1604-2.3.7-ga-cuda9.0_1-1_amd64.deb"
else
  echo "nccl not found"
  exit 2
fi

URL="http://nccl2-deb.cdn.bcebos.com/$DEB"

DIR="/nccl2"
mkdir -p $DIR
# we cached the nccl2 deb package in BOS, so we can download it with wget
# install nccl2: http://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#down
wget -q -O $DIR/$DEB $URL

cd $DIR && ar x $DEB && tar xf data.tar.xz
DEBS=$(find ./var/ -name "*.deb")
for sub_deb in $DEBS; do
  echo $sub_deb
  ar x $sub_deb && tar xf data.tar.xz
done
mv -f usr/include/nccl.h /usr/local/include/
mv -f usr/lib/x86_64-linux-gnu/libnccl* /usr/local/lib/
rm /usr/include/nccl.h
rm -rf $DIR
