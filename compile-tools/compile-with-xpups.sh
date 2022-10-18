#!/bin/bash

git config --global http.sslverify false

rm -rf build
mkdir -p build
cd build
rm -rf *

export PADDLE_VERSION=2.2.1
cp  ../compile-tools/set_http_proxy.sh .

#--user=moonlight --password=mn-12345

sed -i 's|https://pslib.bj.bcebos.com/pslib.tar.gz|ftp://tesla-kt.bcc-szth.baidu.com:8010/paddle-depends/xpups_co_dev/pslib.tar.gz|g' ../cmake/external/pslib.cmake
sed -i 's|wget --no-check-certificate |wget --no-check-certificate --user=moonlight --password=mn-12345 |g' ../cmake/external/pslib.cmake
sed -i 's|BUILD_COMMAND       make -j$(nproc) ${COMMON_ARGS} ${OPTIONAL_ARGS}|BUILD_COMMAND       make -j$(nproc) TARGET=GENERIC ${COMMON_ARGS} ${OPTIONAL_ARGS}|g' ../cmake/external/openblas.cmake

sed -i 's|https://pslib.bj.bcebos.com/pslib_brpc.tar.gz|ftp://tesla-kt.bcc-szth.baidu.com:8010/paddle-depends/xpups_co_dev/pslib_brpc.tar.gz|g' ../cmake/external/pslib_brpc.cmake
sed -i 's|wget --no-check-certificate |wget --no-check-certificate --user=moonlight --password=mn-12345 |g' ../cmake/external/pslib_brpc.cmake
sed -i 's|http://paddlepaddledeps.bj.bcebos.com/|ftp://tesla-kt.bcc-szth.baidu.com:8010/paddle-depends/xpups_co_dev/|g' ../cmake/external/boost.cmake
sed -r -i 's|(   URL  .*)|\1\n    HTTP_USERNAME         moonlight\n    HTTP_PASSWORD         mn-12345|g' ../cmake/external/boost.cmake
sed -i 's|https://paddlepaddledeps.bj.bcebos.com/|ftp://tesla-kt.bcc-szth.baidu.com:8010/paddle-depends/xpups_co_dev/|g' ../cmake/external/lapack.cmake
sed -r -i 's|(   URL  .*)|\1\n    HTTP_USERNAME         moonlight\n    HTTP_PASSWORD         mn-12345|g' ../cmake/external/lapack.cmake
sed -i 's|http://paddlepaddledeps.bj.bcebos.com/|ftp://tesla-kt.bcc-szth.baidu.com:8010/paddle-depends/xpups_co_dev/|g' ../cmake/external/mklml.cmake
sed -r -i 's|(   URL  .*)|\1\n    HTTP_USERNAME         moonlight\n    HTTP_PASSWORD         mn-12345|g' ../cmake/external/mklml.cmake

cmake .. \
    -DCMAKE_INSTALL_PREFIX=./output/ \
    -DWITH_PYTHON=ON \
    -DPY_VERSION=3.7 \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m/ \
    -DPYTHON_LIBRARY=/usr/lib/python3.7/config-3.7m-x86_64-linux-gnu/libpython3.7m.a \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.7 \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_MKL=OFF \
    -DWITH_XPU=ON \
    -DWITH_XPU_BKCL=ON \
    -DWITH_XPU_KP=ON \
    -DWITH_XPU_CACHE_BFID=ON \
    -DWITH_GPU=OFF  \
    -DWITH_FLUID_ONLY=ON \
    -DWITH_DISTRIBUTE=ON \
    -DWITH_GLOO=ON \
    -DWITH_PSLIB=ON \
    -DWITH_PSLIB_BRPC=OFF \
    -DWITH_PSCORE=OFF \
    -DWITH_HETERPS=ON

    #-DGIT_URL="https://github.com.cnpmjs.org" \
    #-DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.7m/ \
    #-DPYTHON_LIBRARY=$PYTHONROOT/lib/libpython3.7m.a \
    #-DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.7 

cd ..
./compile-tools/build-again.sh 0

