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

#include "paddle/phi/kernels/randperm_kernel.h"

#ifdef __NVCC__
#include <curand_kernel.h>
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hiprand_kernel.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/randint_kernel.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/memcpy.h"

DECLARE_bool(use_curand);

namespace phi {

template <typename T>
__global__ void SwapRepeatKernel(
    int* key, T* data, int n, uint64_t seed, uint64_t offset) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < n) return;

  bool first_repeat = false;
  if (data[idx] == data[idx + 1]) {
    if (idx == 0) {
      first_repeat = true;
    } else if (data[idx] != data[idx - 1]) {
      first_repeat = true;
    }
  }

  if (!first_repeat) return;

  int repeat_size = 1;
  for (int i = idx; i < n; ++i) {
    if (data[i] == data[i + 1]) {
      ++repeat_size;
    } else {
      break;
    }
  }

#ifdef __NVCC__
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);
  for (int i = repeat_size - 1; i > 0; i--) {
    uint32_t r = curand(&state) % (i + 1);
#elif __HIPCC__
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, offset, &state);
  for (int i = repeat_size - 1; i > 0; i--) {
    uint32_t r = hiprand(&state) % (i + 1);
#endif
    if (r != i) {
      T tmp = data[idx + i];
      data[idx + i] = data[idx + r];
      data[idx + r] = tmp;
    }
  }
}

template <typename T, typename Context>
void RandpermRawKernel(
    const Context& dev_ctx, int n, DataType dtype, int seed, DenseTensor* out) {
  DenseTensor key;
  RandintKernel<int, Context>(dev_ctx,
                              std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max(),
                              IntArray({n}),
                              phi::DataType::INT32,
                              &key);
  DenseTensor key_out = Empty<int, Context>(dev_ctx, IntArray({n}));

  DenseTensor range = Empty<T, Context>(dev_ctx, IntArray({n}));
  T* range_data = range.data<T>();
  funcs::ForRange<Context> for_range(dev_ctx, n);
  for_range([range_data] __device__(size_t idx) {
    range_data[idx] = static_cast<T>(idx);
  });

  out->Resize(phi::make_ddim({n}));
  T* out_data = dev_ctx.template Alloc<T>(out);

  // Refer to [Algorithm of randperm] https://osf.io/af2hy/ to
  // improve performance of radix sort.
  double n_d = static_cast<double>(n);
  int begin_bit = 0;
  int end_bit =
      std::ceil(std::log2(n_d - (6 * n_d * n_d + 1) / (12 * std::log(0.9))));

  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs<int, T>(nullptr,
                                          temp_storage_bytes,
                                          key.data<int>(),
                                          key_out.data<int>(),
                                          range.data<T>(),
                                          out_data,
                                          n,
                                          begin_bit,
                                          end_bit < 32 ? end_bit : 32,
                                          dev_ctx.stream());

  auto d_temp_storage = paddle::memory::Alloc(dev_ctx, temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs<int, T>(d_temp_storage->ptr(),
                                          temp_storage_bytes,
                                          key.data<int>(),
                                          key_out.data<int>(),
                                          range.data<T>(),
                                          out_data,
                                          n,
                                          begin_bit,
                                          end_bit < 32 ? end_bit : 32,
                                          dev_ctx.stream());

  auto gen_cuda = dev_ctx.GetGenerator();
  auto seed_offset = gen_cuda->IncrementOffset(n);

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n);
  SwapRepeatKernel<T><<<config.block_per_grid.x,
                        config.thread_per_block.x,
                        0,
                        dev_ctx.stream()>>>(
      key_out.data<int>(), out_data, n, seed_offset.first, seed_offset.second);
}

template <typename T, typename Context>
void RandpermKernel(const Context& dev_ctx,
                    int n,
                    DataType dtype,
                    DenseTensor* out) {
  RandpermRawKernel<T>(dev_ctx, n, dtype, 0, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(randperm_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::RandpermRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(randperm,
                   GPU,
                   ALL_LAYOUT,
                   phi::RandpermKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
