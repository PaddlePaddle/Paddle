/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceSum(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T sum_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceMax(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T max_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

}  // namespace operators
}  // namespace paddle