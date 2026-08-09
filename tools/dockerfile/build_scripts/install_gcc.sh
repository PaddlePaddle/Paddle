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

# GCC is installed side-by-side. Callers select its runtime via LD_LIBRARY_PATH.

if [ "$1" == "gcc82" ]; then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/gcc-8.2.0.tar.xz
  tar -xf gcc-8.2.0.tar.xz && \
  cd gcc-8.2.0 && \
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/sanitizer_platform_limits_posix.cc.patch
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/sanitizer_platform_limits_posix.h.patch
  patch -p0 libsanitizer/sanitizer_common/sanitizer_platform_limits_posix.cc sanitizer_platform_limits_posix.cc.patch
  patch -p0 libsanitizer/sanitizer_common/sanitizer_platform_limits_posix.h sanitizer_platform_limits_posix.h.patch
  unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
  ./contrib/download_prerequisites && \
  cd .. && mkdir temp_gcc82 && cd temp_gcc82 && \
  ../gcc-8.2.0/configure --prefix=/usr/local/gcc-8.2 --enable-threads=posix --disable-checking --disable-multilib && \
  make -j8 && make install
  cd .. && rm -rf temp_gcc82 gcc-8.2.0 gcc-8.2.0.tar.xz
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
elif [ "$1" == "gcc121" ]; then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/gcc-12.1.0.tar.gz
  tar -xzf gcc-12.1.0.tar.gz && \
  cd gcc-12.1.0 && \
  unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
  ./contrib/download_prerequisites && \
  cd .. && mkdir temp_gcc121 && cd temp_gcc121 && \
  ../gcc-12.1.0/configure --prefix=/usr/local/gcc-12.1 --enable-checking=release --enable-languages=c,c++ --disable-multilib && \
  make -j8 && make install
  cd .. && rm -rf temp_gcc121 gcc-12.1.0 gcc-12.1.0.tar.gz
elif [ "$1" == "gcc11" ]; then
  GCC_VERSION=${GCC_VERSION:-11.5.0}
  GCC_MAJOR_MINOR=$(echo ${GCC_VERSION} | cut -d. -f1,2)
  GCC_PREFIX=/usr/local/gcc-${GCC_MAJOR_MINOR}
  GCC_ARCHIVE=gcc-${GCC_VERSION}.tar.xz
  wget -q -O "${GCC_ARCHIVE}" "https://xly-devops.bj.bcebos.com/gouzil/GCC%20${GCC_VERSION}.tar.xz"
  tar -xf ${GCC_ARCHIVE} && \
  cd gcc-${GCC_VERSION} && \
  unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
  ./contrib/download_prerequisites && \
  cd .. && mkdir temp_gcc11 && cd temp_gcc11 && \
  ../gcc-${GCC_VERSION}/configure --prefix=${GCC_PREFIX} --enable-checking=release --enable-languages=c,c++ --disable-multilib && \
  make -j$(nproc) && make install
  cd .. && rm -rf temp_gcc11 gcc-${GCC_VERSION} ${GCC_ARCHIVE}
elif [ "$1" == "gcc152" ]; then
  GCC_VERSION=${GCC_VERSION:-15.2.0}
  GCC_MAJOR_MINOR=$(echo ${GCC_VERSION} | cut -d. -f1,2)
  GCC_PREFIX=/usr/local/gcc-${GCC_MAJOR_MINOR}
  GCC_ARCHIVE=gcc-${GCC_VERSION}.tar.xz
  wget -q https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/${GCC_ARCHIVE}
  tar -xf ${GCC_ARCHIVE} && \
  cd gcc-${GCC_VERSION} && \
  unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
  ./contrib/download_prerequisites && \
  cd .. && mkdir temp_gcc152 && cd temp_gcc152 && \
  ../gcc-${GCC_VERSION}/configure --prefix=${GCC_PREFIX} --enable-checking=release --enable-languages=c,c++ --disable-multilib && \
  make -j$(nproc) && make install
  cd .. && rm -rf temp_gcc152 gcc-${GCC_VERSION} ${GCC_ARCHIVE}

fi
