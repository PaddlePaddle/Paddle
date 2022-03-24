// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

template <typename T, size_t Rank>
__global__ void RollCudaKernel(const T* input,
                               T* output,
                               int64_t N,
                               phi::Array<int64_t, Rank> shifts,
                               phi::Array<int64_t, Rank> strides,
                               phi::Array<int64_t, Rank> sizes) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int64_t output_idx = idx;
  int64_t new_dim_idx = 0;

#pragma unroll
  for (size_t i = 0; i < Rank; i++) {
    new_dim_idx = (idx / strides[i]) % sizes[i] + shifts[i];
    if (new_dim_idx >= sizes[i]) {
      output_idx += (shifts[i] - sizes[i]) * strides[i];
    } else {
      output_idx += shifts[i] * strides[i];
    }
  }
  output[output_idx] = input[idx];
}

#define CALL_ROLL_CUDA_KERNEL(N)                                              \
  case N: {                                                                   \
    phi::Array<int64_t, N> _strides;                                          \
    phi::Array<int64_t, N> _shifts;                                           \
    phi::Array<int64_t, N> _sizes;                                            \
    for (size_t idx = 0; idx < N; ++idx) {                                    \
      _strides[idx] = strides[idx];                                           \
      _shifts[idx] = shifts_data[idx];                                        \
      _sizes[idx] = sizes[idx];                                               \
    }                                                                         \
    RollCudaKernel<                                                           \
        T,                                                                    \
        N><<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS, \
             PADDLE_CUDA_NUM_THREADS,                                         \
             0,                                                               \
             stream>>>(in_data, out_data, numel, _shifts, _strides, _sizes);  \
    break;                                                                    \
  }

}  // namespace phi
