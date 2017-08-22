#!/bin/bash

set -xe

mkdir -p /paddle/build_android/$ANDROID_ABI
cd /paddle/build_android/$ANDROID_ABI
rm -rf /paddle/install 2>/dev/null || true

THIRD_PARTY_PATH=/paddle/third_party_android/$ANDROID_ABI

if [ $ANDROID_ABI == "armeabi-v7a" ]; then
  cmake -DCMAKE_SYSTEM_NAME=Android \
        -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_ARM_STANDALONE_TOOLCHAIN \
        -DANDROID_ABI=$ANDROID_ABI \
        -DANDROID_ARM_NEON=ON \
        -DANDROID_ARM_MODE=ON \
        -DHOST_C_COMPILER=/usr/bin/gcc \
        -DHOST_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_INSTALL_PREFIX=/paddle/install \
        -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_C_API=ON \
        -DWITH_SWIG_PY=OFF \
        /paddle
elif [ $ANDROID_ABI == "arm64-v7a" ]; then
  cmake -DCMAKE_SYSTEM_NAME=Android \
        -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_ARM64_STANDALONE_TOOLCHAIN \
        -DANDROID_ABI=$ANDROID_ABI \
        -DANDROID_ARM_MODE=ON \
        -DHOST_C_COMPILER=/usr/bin/gcc \
        -DHOST_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_INSTALL_PREFIX=/paddle/install \
        -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_C_API=ON \
        -DWITH_SWIG_PY=OFF \
        /paddle
elif [ $ANDROID_ABI == "armeabi" ]; then
  cmake -DCMAKE_SYSTEM_NAME=Android \
        -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_ARM_STANDALONE_TOOLCHAIN \
        -DANDROID_ABI=$ANDROID_ABI \
        -DANDROID_ARM_MODE=ON \
        -DHOST_C_COMPILER=/usr/bin/gcc \
        -DHOST_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_INSTALL_PREFIX=/paddle/install \
        -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_C_API=ON \
        -DWITH_SWIG_PY=OFF \
        /paddle
else
  echo "Invalid ANDROID_ABI: $ANDROID_ABI"
fi

make -j `nproc`
make install -j `nproc`
