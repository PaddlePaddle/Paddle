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

# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -ex

if [ -f "/etc/redhat-release" ];then
  lib_so_5=/usr/lib64/libgfortran.so.5
  lib_so_6=/usr/lib64/libstdc++.so.6
  lib_path=/usr/lib64
else
  lib_so_5=/usr/lib/x86_64-linux-gnu/libstdc++.so.5
  lib_so_6=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  lib_path=/usr/lib/x86_64-linux-gnu
fi

if [ "$1" == "gcc82" ]; then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/gcc-8.2.0.tar.xz 
  tar -xvf gcc-8.2.0.tar.xz && \
  cd gcc-8.2.0 && \
  unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
  ./contrib/download_prerequisites && \
  cd .. && mkdir temp_gcc82 && cd temp_gcc82 && \
  ../gcc-8.2.0/configure --prefix=/usr/local/gcc-8.2 --enable-threads=posix --disable-checking --disable-multilib && \
  make -j8 && make install
  cd .. && rm -rf temp_gcc82 gcc-8.2.0 gcc-8.2.0.tar.xz
  cp ${lib_so_6} ${lib_so_6}.bak  && rm -f ${lib_so_6} && 
  ln -s /usr/local/gcc-8.2/lib64/libgfortran.so.5 ${lib_so_5} && \
  ln -s /usr/local/gcc-8.2/lib64/libstdc++.so.6 ${lib_so_6} && \
  cp /usr/local/gcc-8.2/lib64/libstdc++.so.6.0.25 ${lib_path}
elif [ "$1" == "gcc54" ]; then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/gcc-5.4.0.tar.gz
  tar -xzf gcc-5.4.0.tar.gz && \
  cd gcc-5.4.0 && \
  unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
  ./contrib/download_prerequisites && \
  cd .. && mkdir temp_gcc54 && cd temp_gcc54 && \
  ../gcc-5.4.0/configure --prefix=/usr/local/gcc-5.4 --enable-checking=release --enable-languages=c,c++ --disable-multilib && \
  make -j8 && make install
  cd .. && rm -rf temp_gcc54 gcc-5.4.0 gcc-5.4.0.tar.gz
  cp ${lib_so_6} ${lib_so_6}.bak  && rm -f ${lib_so_6} && 
  ln -s /usr/local/gcc-5.4/lib64/libgfortran.so.5 ${lib_so_5} && \
  ln -s /usr/local/gcc-5.4/lib64/libstdc++.so.6 ${lib_so_6} && \
  cp /usr/local/gcc-5.4/lib64/libstdc++.so.6.0.21 ${lib_path}
elif [ "$1" == "gcc122" ]; then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/gcc-12.2.0.tar.gz
  tar -xzf gcc-12.2.0.tar.gz && \
  cd gcc-12.2.0 && \
  unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
  ./contrib/download_prerequisites && \
  cd .. && mkdir temp_gcc122 && cd temp_gcc122 && \
  ../gcc-12.2.0/configure --prefix=/usr/local/gcc-12.2 --enable-checking=release --enable-languages=c,c++ --disable-multilib && \
  make -j8 && make install
  cd .. && rm -rf temp_gcc122 gcc-12.2.0 gcc-12.2.0.tar.gz
  cp ${lib_so_6} ${lib_so_6}.bak  && rm -f ${lib_so_6} &&
  ln -s /usr/local/gcc-12.2/lib64/libgfortran.so.5 ${lib_so_5} && \
  ln -s /usr/local/gcc-12.2/lib64/libstdc++.so.6 ${lib_so_6} && \
  cp /usr/local/gcc-12.2/lib64/libstdc++.so.6.0.30 ${lib_path}
fi
