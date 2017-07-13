#!/bin/bash
set -e

# Create the build directory for CMake.
mkdir -p $TRAVIS_BUILD_DIR/build_android
cd $TRAVIS_BUILD_DIR/build_android

# Compile paddle binaries
cmake -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_STANDALONE_TOOLCHAIN=$HOME/android-toolchain-gcc \
      -DANDROID_ABI=armeabi-v7a \
      -DANDROID_ARM_NEON=ON \
      -DANDROID_ARM_MODE=ON \
      -DWITH_C_API=ON \
      -DWITH_SWIG_PY=OFF \
      -DWITH_STYLE_CHECK=OFF \
      ..

make -j `nproc`
