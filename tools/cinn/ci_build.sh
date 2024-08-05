#!/bin/bash

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

readonly workspace=$PWD

function install_isl {
    cd $workspace
    if [ ! -d isl ]; then
        git clone https://github.com/inducer/isl.git isl
    fi

    cd isl
    git checkout a72ac2e
    ./autogen.sh

    find /usr -name "SourceLocation.h"

    CFLAGS="-fPIC -DPIC" CPPFLAGS="-fPIC -DPIC" ./configure --with-clang=system --enable-shared=yes --enable-static=yes
    make -j install
    cd $workspace
}

function install_ginac {
    cd $workspace
    if [ ! -d gmp-6.2.1 ]; then
      wget https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz
      tar xf gmp-6.2.1.tar.xz
      cd gmp-6.2.1
      CFLAGS="-fPIC -DPIC" CXXFLAGS="-fPIC -DPIC" ./configure --enable-shared=yes --enable-static=yes
      make -j install
    fi

    if [ ! -d cln-1.3.6 ]; then
      wget https://www.ginac.de/CLN/cln-1.3.6.tar.bz2 -O cln-1.3.6.tar.bz2
      tar xf cln-1.3.6.tar.bz2
      cd cln-1.3.6
      CFLAGS="-fPIC -DPIC" CXXFLAGS="-fPIC -DPIC" ./configure --enable-shared=yes --enable-static=yes --with-gmp=/usr/local
      make -j install
    fi

    if [ ! -d ginac-1.8.1 ]; then
      wget https://www.ginac.de/ginac-1.8.1.tar.bz2 -O ginac-1.8.1.tar.bz2
      tar xf ginac-1.8.1.tar.bz2
      cd ginac-1.8.1
      CFLAGS="-fPIC -DPIC" CXXFLAGS="-fPIC -DPIC" CLN_LIBS="-L/usr/local/lib -lcln" CLN_CFLAGS="-I/usr/local/include" ./configure --enable-shared=yes --enable-static=yes
      make -j install
    fi

    cd $workspace
}

function compile_cinn {
    cd $workspace
    cmake .
    make -j
}

function run_test {
    ctest -V
}

#install_isl
#install_ginac
#
#compile_cinn

#run_test
