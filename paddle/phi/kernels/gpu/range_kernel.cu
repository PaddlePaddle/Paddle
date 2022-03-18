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

#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/range_kernel.h"

namespace phi {

template <typename T>
__global__ void Range(T start, T step, int64_t size, T* out) {
  CUDA_KERNEL_LOOP(index, size) { out[index] = start + step * index; }
}

template <typename T, typename Context>
void RangeKernel(const Context& dev_ctx,
                 const DenseTensor& start,
                 const DenseTensor& end,
                 const DenseTensor& step,
                 DenseTensor* out) {
  T start_value = GetValue<T>(start);
  T end_value = GetValue<T>(end);
  T step_value = GetValue<T>(step);

  int64_t size = 0;
  GetSize(start_value, end_value, step_value, &size);
  out->Resize(phi::make_ddim({size}));
  T* out_data = dev_ctx.template Alloc<T>(out);

  int block = std::min(size, static_cast<int64_t>(256));
  int grid = (size + block - 1) / block;
  Range<T><<<grid, block, 0, dev_ctx.stream()>>>(
      start_value, step_value, size, out_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    range, GPU, ALL_LAYOUT, phi::RangeKernel, float, double, int64_t, int) {}
