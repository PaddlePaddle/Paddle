#!/bin/bash
set -e

# Create the build directory for CMake.
mkdir -p $TRAVIS_BUILD_DIR/build_ios
cd $TRAVIS_BUILD_DIR/build_ios

# Compile paddle binaries
cmake -DCMAKE_SYSTEM_NAME=iOS \
      -DIOS_PLATFORM=OS \
      -DWITH_C_API=ON \
      -DWITH_TESTING=OFF \
      -DWITH_SWIG_PY=OFF \
      -DWITH_STYLE_CHECK=OFF \
      ..

make -j `nproc`
