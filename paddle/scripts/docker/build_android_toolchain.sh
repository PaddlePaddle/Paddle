#!/bin/bash

# This file is supposed to be called from Dockerfile.android.

set -xe

if [ $ANDROID_ABI == "arm64-v8a" ]; then
  ANDROID_ARCH=arm64
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

