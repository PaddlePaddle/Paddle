#!/bin/bash
set -e

# Create the build directory for CMake.
mkdir -p $TRAVIS_BUILD_DIR/build_ios
cd $TRAVIS_BUILD_DIR/build_ios

# Compile paddle binaries
cmake -DCMAKE_SYSTEM_NAME=iOS \
      -DIOS_PLATFORM=OS \
      -DCMAKE_OSX_ARCHITECTURES="arm64" \
      -DWITH_C_API=ON \
      -DUSE_EIGEN_FOR_BLAS=ON \
      -DWITH_TESTING=OFF \
      -DWITH_SWIG_PY=OFF \
      -DWITH_STYLE_CHECK=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      ..

make -j 2
