#!/bin/bash

set -xe

COMPILER=gcc
USE_EIGEN=ON
if [ $COMPILER == clang ]; then
  SUFFIX=_clang
  C_COMPILER=clang
  CXX_COMPILER=clang++
else
  SUFFIX=_gcc
  C_COMPILER=gcc
  CXX_COMPILER=g++
fi
if [ $USE_EIGEN == ON ]; then
  SUFFIX=${SUFFIX}_eigen
else
  SUFFIX=${SUFFIX}_openblas
fi

BUILD_ROOT=/paddle/build_android$SUFFIX
DEST_ROOT=/paddle/install$SUFFIX

rm -rf $BUILD_ROOT 2>/dev/null || true
mkdir -p $BUILD_ROOT
cd $BUILD_ROOT

THIRD_PARTY_PATH=/paddle/third_party_android$SUFFIX/$ANDROID_ABI

if [ $ANDROID_ABI == "armeabi-v7a" ]; then
  cmake -DCMAKE_SYSTEM_NAME=Android \
        -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_ARM_STANDALONE_TOOLCHAIN \
        -DANDROID_ABI=$ANDROID_ABI \
        -DANDROID_ARM_NEON=ON \
        -DANDROID_ARM_MODE=ON \
        -DCMAKE_C_COMPILER=$ANDROID_ARM_STANDALONE_TOOLCHAIN/bin/arm-linux-androideabi-${C_COMPILER} \
        -DCMAKE_CXX_COMPILER=$ANDROID_ARM_STANDALONE_TOOLCHAIN/bin/arm-linux-androideabi-${CXX_COMPILER} \
        -DHOST_C_COMPILER=/usr/bin/gcc \
        -DHOST_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_EIGEN_FOR_BLAS=${USE_EIGEN} \
        -DWITH_C_API=ON \
        -DWITH_SWIG_PY=OFF \
        -DWITH_STYLE_CHECK=OFF \
        ..
elif [ $ANDROID_ABI == "arm64-v8a" ]; then
  cmake -DCMAKE_SYSTEM_NAME=Android \
        -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_ARM64_STANDALONE_TOOLCHAIN \
        -DANDROID_ABI=$ANDROID_ABI \
        -DANDROID_ARM_MODE=ON \
        -DCMAKE_C_COMPILER=$ANDROID_ARM64_STANDALONE_TOOLCHAIN/bin/aarch64-linux-android-${C_COMPILER} \
        -DCMAKE_CXX_COMPILER=$ANDROID_ARM64_STANDALONE_TOOLCHAIN/bin/aarch64-linux-android-${CXX_COMPILER} \
        -DHOST_C_COMPILER=/usr/bin/gcc \
        -DHOST_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_EIGEN_FOR_BLAS=${USE_EIGEN} \
        -DWITH_C_API=ON \
        -DWITH_SWIG_PY=OFF \
        ..
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
        ..
else
  echo "Invalid ANDROID_ABI: $ANDROID_ABI"
fi

make VERBOSE=1 -j2
make install -j2
