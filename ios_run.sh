#!/bin/bash

set -xe

mkdir -p ./ios_build
cd ./ios_build

cmake -DCMAKE_SYSTEM_NAME=Darwin \
					   -DWITH_C_API=ON \
					   -DWITH_TESTING=OFF \
					   -DWITH_SWIG_PY=OFF \
					   -DCMAKE_BUILD_TYPE=Release \
					   -DCMAKE_INSTALL_PREFIX=/Users/xingzhaolong/cross_compile/ios \
					   ..
                      # -DIOS_PLATFORM=SIMULATOR \
                       #-DCMAKE_Go_COMPILER=/usr/local/bin \

