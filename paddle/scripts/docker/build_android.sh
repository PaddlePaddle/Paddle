#!/bin/bash

set -xe

if [ $ANDROID_ABI == "arm64-v8a" ]; then
  ANDROID_ARCH=arm64
  if [ $ANDROID_API -lt 21 ]; then
    echo "Warning: arm64-v8a requires ANDROID_API >= 21."
    ANDROID_API=21
  fi
else # armeabi, armeabi-v7a
  ANDROID_ARCH=arm
fi

ANDROID_STANDALONE_TOOLCHAIN=$ANDROID_TOOLCHAINS_DIR/$ANDROID_ARCH-android-$ANDROID_API

cat <<EOF
============================================
Generating the standalone toolchain ...
${ANDROID_NDK_HOME}/build/tools/make-standalone-toolchain.sh
      --arch=$ANDROID_ARCH
      --platform=android-$ANDROID_API
      --install-dir=${ANDROID_STANDALONE_TOOLCHAIN}
============================================
EOF
${ANDROID_NDK_HOME}/build/tools/make-standalone-toolchain.sh \
      --arch=$ANDROID_ARCH \
      --platform=android-$ANDROID_API \
      --install-dir=$ANDROID_STANDALONE_TOOLCHAIN

BUILD_ROOT=/paddle/build_android
DEST_ROOT=/paddle/install_android

rm -rf $BUILD_ROOT 2>/dev/null || true
mkdir -p $BUILD_ROOT
cd $BUILD_ROOT

if [ $ANDROID_ABI == "armeabi-v7a" ]; then
  cmake -DCMAKE_SYSTEM_NAME=Android \
        -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_STANDALONE_TOOLCHAIN \
        -DANDROID_ABI=$ANDROID_ABI \
        -DANDROID_ARM_NEON=ON \
        -DANDROID_ARM_MODE=ON \
        -DHOST_C_COMPILER=/usr/bin/gcc \
        -DHOST_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_EIGEN_FOR_BLAS=ON \
        -DWITH_C_API=ON \
        -DWITH_SWIG_PY=OFF \
        -DWITH_STYLE_CHECK=OFF \
        ..
elif [ $ANDROID_ABI == "arm64-v8a" ]; then
  cmake -DCMAKE_SYSTEM_NAME=Android \
        -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_STANDALONE_TOOLCHAIN \
        -DANDROID_ABI=$ANDROID_ABI \
        -DANDROID_ARM_MODE=ON \
        -DHOST_C_COMPILER=/usr/bin/gcc \
        -DHOST_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_EIGEN_FOR_BLAS=OFF \
        -DWITH_C_API=ON \
        -DWITH_SWIG_PY=OFF \
        -DWITH_STYLE_CHECK=OFF \
        ..
elif [ $ANDROID_ABI == "armeabi" ]; then
  cmake -DCMAKE_SYSTEM_NAME=Android \
        -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_STANDALONE_TOOLCHAIN \
        -DANDROID_ABI=$ANDROID_ABI \
        -DANDROID_ARM_MODE=ON \
        -DHOST_C_COMPILER=/usr/bin/gcc \
        -DHOST_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_C_API=ON \
        -DWITH_SWIG_PY=OFF \
        -DWITH_STYLE_CHECK=OFF \
        ..
else
  echo "Invalid ANDROID_ABI: $ANDROID_ABI"
fi

cat <<EOF
============================================
Building in $BUILD_ROOT ...
============================================
EOF
make -j `nproc`
make install -j `nproc`
