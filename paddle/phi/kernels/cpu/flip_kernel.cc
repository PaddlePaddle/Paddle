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

#include "paddle/phi/kernels/flip_kernel.h"

#include <bitset>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

constexpr size_t dim_bitset_size = 64;

template <typename T, typename Context>
void FlipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int>& axis,
                DenseTensor* out) {
  auto x_dims = x.dims();
  const int total_dims = x_dims.size();
  std::bitset<dim_bitset_size> dim_bitset;
  for (auto& item : axis) {
    auto dim = item;
    if (item < 0) {
      dim += total_dims;
    }
    dim_bitset[dim] = true;
  }
  auto x_strides = common::stride(x_dims);
  auto numel = x.numel();
  const T* x_data = x.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < numel; ++i) {
    int64_t cur_indices = i;
    int64_t rem = 0;
    int64_t dst_offset = 0;

    for (int d = 0; d < total_dims; ++d) {
      int64_t temp = cur_indices;
      cur_indices = cur_indices / x_strides[d];
      rem = temp - cur_indices * x_strides[d];
      dst_offset += dim_bitset[d] ? (x_dims[d] - 1 - cur_indices) * x_strides[d]
                                  : cur_indices * x_strides[d];
      cur_indices = rem;
    }
    out_data[i] = x_data[dst_offset];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(flip,
                   CPU,
                   ALL_LAYOUT,
                   phi::FlipKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
