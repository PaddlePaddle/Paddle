/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <glog/logging.h>
#include "paddle/fluid/operators/math.h"

namespace paddle {
namespace operators {

#if defined(__CUDACC__) && CUDA_VERSION >= 7050

#define DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Func, expr, FP16Function)            \
  template <typename T>                                                       \
  __global__ void SameDimsElemwise##Func##CUDAKernel(const T* x, const T* y,  \
                                                     T* z, int64_t size) {    \
    int col = blockIdx.x * blockDim.x + threadIdx.x;                          \
    while (col < size) {                                                      \
      z[col] = x[col] expr y[col];                                            \
      col += blockDim.x * gridDim.x;                                          \
    }                                                                         \
  }                                                                           \
  template <>                                                                 \
  inline __global__ void SameDimsElemwise##Func##CUDAKernel<half>(            \
      const half* x, const half* y, half* z, int64_t size) {                  \
    int start = threadIdx.x + blockDim.x * blockIdx.x;                        \
    int stride = blockDim.x * gridDim.x;                                      \
    int n2 = size / 2;                                                        \
    const half2* x2 = reinterpret_cast<const half2*>(x);                      \
    const half2* y2 = reinterpret_cast<const half2*>(y);                      \
    half2* z2 = reinterpret_cast<half2*>(z);                                  \
    for (int i = start; i < n2; i += stride) {                                \
      z2[i] = FP16Function(x2[i], y2[i]);                                     \
    }                                                                         \
    if (start == 0 && (size % 2)) z[size - 1] = x[size - 1] expr y[size - 1]; \
  }
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Add, +, half2_add)
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Sub, -, half2_sub)
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Mul, *, half2_mul)
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Div, /, half2_div)
#undef DEFINE_SIMPLE_CUDA_BINARY_KERNEL

#endif  // PADDLE_CUDA_FP16

}  // namespace operators
}  // namespace paddle
