#!/usr/bin/env bash

#rm -rf build.lite.android.arm64-v8a/C* \
#  build.lite.android.arm64-v8a/Makefile \
#  build.lite.android.arm64-v8a/cmake_install.cmake \
#  build.lite.android.arm64-v8a/paddle

./paddle/fluid/lite/tools/build.sh build_test_arm

echo "adb push /Users/yuanshuai06/Baidu/code/Paddle/lite-mobile/build/paddle/fluid/lite/kernels/arm/test_mul_compute_arm /data/local/tmp/lite"

echo "adb shell "/data/local/tmp/lite/test_mul_compute_arm""
