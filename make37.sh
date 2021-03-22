#!/bin/bash
(($#!=1)) && echo "usage $0 all/fast" && exit -1
export http_proxy="http://172.19.57.45:3128"
export https_proxy="http://172.19.57.45:3128"

CWD=$PWD/..

if [ $1 != "fast" ];then
rm -rf build/ && mkdir build && cd build
cmake .. \
-DWITH_MKL=OFF \
-DWITH_FLUID_ONLY=ON \
-DCUDA_ARCH_NAME=Volta \
-DPYTHON_EXECUTABLE=$CWD/py37/bin/python \
-DWITH_GLOO=ON \
-DWITH_DISTRIBUTE=ON \
-DPYTHON_INCLUDE_DIR=$CWD/py37/include/python3.7m/ \
-DPYTHON_LIBRARY=$CWD/py37/lib/python3.7/\
-DPY_VERSION=3.7 \
-DWITH_GPU=ON \
-DWITH_TESTING=OFF \
-DCMAKE_BUILD_TYPE=Release \

else
cd build
fi

make -j 40
