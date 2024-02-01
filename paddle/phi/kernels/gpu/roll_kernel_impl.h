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

#include "paddle/common/array.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void RollCudaKernel(const T* input,
                               T* output,
                               const int rank,
                               const int64_t numel,
                               phi::Array<int64_t, DDim::kMaxRank> shifts,
                               phi::Array<int64_t, DDim::kMaxRank> strides,
                               phi::Array<int64_t, DDim::kMaxRank> sizes) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  if (idx >= numel) {
    return;
  }

  int64_t output_idx = idx;
  int64_t new_dim_idx = 0;

#pragma unroll
  for (size_t i = 0; i < DDim::kMaxRank; i++) {
    if (i >= rank) {
      break;
    }
    new_dim_idx = (output_idx / strides[i]) % sizes[i] + shifts[i];
    if (new_dim_idx >= sizes[i]) {
      output_idx += (shifts[i] - sizes[i]) * strides[i];
    } else {
      output_idx += shifts[i] * strides[i];
    }
  }
  output[output_idx] = input[idx];
}

template <typename T, typename Context>
void LaunchRollKernel(const Context& dev_ctx,
                      const T* input,
                      T* output,
                      const int rank,
                      const int64_t numel,
                      const std::vector<int64_t> shifts,
                      const std::vector<int64_t> strides,
                      const std::vector<int64_t> sizes) {
  using phi::PADDLE_CUDA_NUM_THREADS;

  phi::Array<int64_t, DDim::kMaxRank> strides_array;
  phi::Array<int64_t, DDim::kMaxRank> shifts_array;
  phi::Array<int64_t, DDim::kMaxRank> sizes_array;
  for (int i = 0; i < rank; ++i) {
    strides_array[i] = strides[i];
    shifts_array[i] = shifts[i];
    sizes_array[i] = sizes[i];
  }

  auto stream = dev_ctx.stream();
  RollCudaKernel<T>
      <<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
         PADDLE_CUDA_NUM_THREADS,
         0,
         stream>>>(
          input, output, rank, numel, shifts_array, strides_array, sizes_array);
}

}  // namespace phi
