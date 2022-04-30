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

#include "paddle/phi/kernels/nanmedian_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void NanmedianGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& medians,
                         const DenseTensor& out_grad,
                         DenseTensor* x_grad) {
  const T* x_ptr = x.data<T>();
  const T* m_ptr = medians.data<T>();
  const T* out_grad_ptr = out_grad.data<T>();

  int64_t numel = x.numel();
  auto x_dim = x.dims();
  int64_t x_rank = x_dim.size();
  int64_t stride = x_dim[x_rank - 1];
  auto zero = static_cast<T>(0);

  if (x_grad) {
    T* x_grad_ptr = dev_ctx.template Alloc<T>(x_grad);
    int64_t i = 0;
    for (i = 0; i < numel; i++) {
      if (std::isnan(static_cast<double>(x_ptr[i]))) {
        x_grad_ptr[i] = zero;
        continue;
      }

      int64_t row = static_cast<int64_t>(i / stride);
      int64_t m_row = 2 * row;
      if (std::isnan(static_cast<double>(m_ptr[m_row])) ||
          (x_ptr[i] != m_ptr[m_row] && x_ptr[i] != m_ptr[m_row + 1])) {
        x_grad_ptr[i] = zero;
        continue;
      }

      x_grad_ptr[i] = out_grad_ptr[row];
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(nanmedian_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::NanmedianGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
