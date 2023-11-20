// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/number_count_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
#define CEIL(_x_, _y_) (((_x_)-1) / (_y_) + 1)
#define PERTHREAD_EXPERTS 256
#define WARP_SIZE 32

const int CUDA_NUM_THREADS = 512;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void initialize_zero_kernel(T* data, const int length) {
  CUDA_KERNEL_LOOP(idx, length) { data[idx] = static_cast<T>(0); }
}

template <typename T>
__global__ void NumberCount(const T* numbers,
                            T* number_count,
                            int64_t batch_size,
                            int upper_range) {
  int res_tmp[PERTHREAD_EXPERTS] = {0};
  int expert_min = blockIdx.x * PERTHREAD_EXPERTS;
  int expert_max = expert_min + PERTHREAD_EXPERTS;
  if (expert_max > upper_range) {
    expert_max = upper_range;
  }
  for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
    T idx = numbers[i];
    if (idx == -1) {
      continue;
    }
    if (idx < expert_min || idx >= expert_max) {
      continue;
    }
    res_tmp[idx - expert_min] += 1;
  }
  for (int i = expert_min; i < expert_max; ++i) {
    int x = res_tmp[i - expert_min];
#pragma unroll
    for (int j = 1; j < WARP_SIZE; j <<= 1) {
#ifdef __HIPCC__
      x = x + __shfl_down(x, j);
#else
      x = x + __shfl_down_sync(-1u, x, j);
#endif
    }
    if (threadIdx.x % WARP_SIZE == 0) {
      phi::CudaAtomicAdd(number_count + i, x);
    }
  }
}

template <typename T, typename Context>
void NumberCountKernel(const Context& ctx,
                       const DenseTensor& numbers,
                       int upper_range,
                       DenseTensor* out) {
  int64_t batch_size = numbers.numel();

  DDim out_dims = common::make_ddim({upper_range});
  out->Resize(out_dims);
  auto out_data = ctx.template Alloc<T>(out);
  const T* gate_data = numbers.data<T>();

  initialize_zero_kernel<T>
      <<<GET_BLOCKS(upper_range), CUDA_NUM_THREADS, 0, ctx.stream()>>>(
          out_data, upper_range);

  NumberCount<T>
      <<<CEIL(upper_range, PERTHREAD_EXPERTS), 256, 0, ctx.stream()>>>(
          gate_data, out_data, batch_size, upper_range);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    number_count, GPU, ALL_LAYOUT, phi::NumberCountKernel, int64_t) {}
