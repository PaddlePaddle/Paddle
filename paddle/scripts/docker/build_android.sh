#!/bin/bash

set -xe

mkdir -p /paddle/build
cd /paddle/build
rm -f /paddle/install 2>/dev/null || true
cmake -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_STANDALONE_TOOLCHAIN \
      -DANDROID_ABI=armeabi-v7a \
      -DANDROID_ARM_NEON=ON \
      -DANDROID_ARM_MODE=ON \
      -DHOST_C_COMPILER=/usr/bin/gcc \
      -DHOST_CXX_COMPILER=/usr/bin/g++ \
      -DCMAKE_INSTALL_PREFIX=/paddle/install \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O3" \
      -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3" \
      -DWITH_C_API=ON \
      -DWITH_SWIG_PY=OFF \
      ..
make -j `nproc`
make install

export PATH=/paddle/install/bin:/paddle/install/opt/paddle/bin:$PATH
paddle version
