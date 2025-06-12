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

#include "paddle/phi/kernels/where_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
namespace phi {

template <typename T, typename Context>
void WhereGradKernel(const Context& dev_ctx,
                     const DenseTensor& condition,
                     const DenseTensor& x UNUSED,
                     const DenseTensor& y UNUSED,
                     const DenseTensor& out_grad,
                     DenseTensor* x_grad,
                     DenseTensor* y_grad) {
  const auto* cond_data = condition.data<bool>();
  auto numel = condition.numel();
  auto* dout = out_grad.data<T>();
  if (out_grad.numel() == 0) {
    if (x_grad) {
      phi::Full<T, Context>(dev_ctx,
                            phi::IntArray(common::vectorize(x_grad->dims())),
                            static_cast<T>(0),
                            x_grad);
    }
    if (y_grad) {
      phi::Full<T, Context>(dev_ctx,
                            phi::IntArray(common::vectorize(y_grad->dims())),
                            static_cast<T>(0),
                            y_grad);
    }
    return;
  }

  if (x_grad != nullptr) {
    auto* dx = dev_ctx.template Alloc<T>(x_grad);
    for (int i = 0; i < numel; i++) {
      dx[i] = cond_data[i] ? dout[i] : T{};
    }
  }
  if (y_grad != nullptr) {
    auto* dy = dev_ctx.template Alloc<T>(y_grad);
    for (int i = 0; i < numel; i++) {
      dy[i] = cond_data[i] ? T{} : dout[i];
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(where_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::WhereGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
