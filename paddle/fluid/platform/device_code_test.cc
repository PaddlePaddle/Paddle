/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device_code.h"
#include "gtest/gtest.h"

constexpr auto saxpy_code = R"(
__global__ void saxpy_kernel(float a, float *x, float* y, float* z, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    z[tid] = a * x[tid] + y[tid];
  }
}
)";

#ifdef PADDLE_WITH_CUDA
TEST(device_code, cuda) {
  paddle::platform::CUDADeviceCode code("saxpy_kernel", saxpy_code, 70);
  code.Compile();
}
#endif
