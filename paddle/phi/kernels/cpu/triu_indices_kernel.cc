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

#include "paddle/phi/kernels/triu_indices_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void TriuIndicesKernel(const Context& dev_ctx,
                       int row,
                       int col,
                       int offset,
                       DataType dtype,
                       DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  const auto& out_dims = out->dims();
  int64_t triu_size = out_dims[1];
  int64_t i = 0;
  T c = std::max<int64_t>(0, offset), r = 0;
  while (i < triu_size) {
    out_data[i] = r;
    out_data[triu_size + i++] = c;

    // move to the next column and check if (r, c) is still in bound
    c += 1;
    if (c >= col) {
      r += 1;
      // not typing std::max with scalar_t as it could be an unsigned type
      // NOTE: not necessary to check if c is less than col or overflows here,
      // because i and triu_size act as a guard.
      c = std::max<int64_t>(0, r + offset);
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    triu_indices, CPU, ALL_LAYOUT, phi::TriuIndicesKernel, int, int64_t) {}
