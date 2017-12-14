#!/bin/bash
set -e

ANDROID_STANDALONE_TOOLCHAIN=$HOME/android-toolchain-gcc
TMP_DIR=$HOME/$JOB/tmp
mkdir -p $TMP_DIR
cd $TMP_DIR
wget -q https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip
unzip -q android-ndk-r14b-linux-x86_64.zip
chmod +x $TMP_DIR/android-ndk-r14b/build/tools/make-standalone-toolchain.sh
$TMP_DIR/android-ndk-r14b/build/tools/make-standalone-toolchain.sh --force --arch=arm --platform=android-21 --install-dir=$ANDROID_STANDALONE_TOOLCHAIN
cd $HOME
rm -rf $TMP_DIR

# Create the build directory for CMake.
mkdir -p $TRAVIS_BUILD_DIR/build_android
cd $TRAVIS_BUILD_DIR/build_android

# Compile paddle binaries
cmake -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_STANDALONE_TOOLCHAIN \
      -DANDROID_ABI=armeabi-v7a \
      -DANDROID_ARM_NEON=ON \
      -DANDROID_ARM_MODE=ON \
      -DUSE_EIGEN_FOR_BLAS=ON \
      -DWITH_C_API=ON \
      -DWITH_SWIG_PY=OFF \
      -DWITH_STYLE_CHECK=OFF \
      ..

make -j `nproc`
