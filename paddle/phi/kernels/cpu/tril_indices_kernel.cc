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

#include "paddle/phi/kernels/tril_indices_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void TrilIndicesKernel(const Context& dev_ctx,
                       int rows,
                       int cols,
                       int offset,
                       DataType dtype,
                       DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  const auto& out_dims = out->dims();
  int64_t tril_size = out_dims[1];
  int64_t i = 0;
  T r = std::max<int64_t>(0, -offset), c = 0;
  while (i < tril_size) {
    out_data[i] = r;
    out_data[tril_size + i++] = c;

    // move to the next column and check if (r, c) is still in bound
    c += 1;
    if (c > r + offset || c >= cols) {
      r += 1;
      c = 0;
      // NOTE: not necessary to check if r is less than row here, because i
      // and tril_size provide the guarantee
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    tril_indices, CPU, ALL_LAYOUT, phi::TrilIndicesKernel, int, int64_t) {}
