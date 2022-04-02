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

#include "paddle/phi/kernels/diag_grad_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/diag_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void DiagGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    int offset,
                    DenseTensor* x_grad) {
  T* dx_data = dev_ctx.template Alloc<T>(x_grad);
  const T* dout_data = out_grad.data<T>();
  auto dx_dims = x_grad->dims();
  auto dout_dims = out_grad.dims();

  if (dx_dims.size() == 1) {
    auto dx_length = dx_dims[0];
    int dx_stride = phi::funcs::ComputeStride(0, dx_dims);

    auto dout_stride_0 = phi::funcs::ComputeStride(0, dout_dims);
    auto dout_stride_1 = phi::funcs::ComputeStride(1, dout_dims);
    dout_data +=
        (offset >= 0 ? offset * dout_stride_1 : -offset * dout_stride_0);

    for (int i = 0; i < dx_length; i++) {
      dx_data[i * dx_stride] = dout_data[i * (dout_stride_0 + dout_stride_1)];
    }
  } else {
    phi::funcs::SetConstant<Context, T> set_padding_value;
    set_padding_value(dev_ctx, x_grad, static_cast<T>(0));

    int dx_stride_0 = phi::funcs::ComputeStride(0, dx_dims);
    int dx_stride_1 = phi::funcs::ComputeStride(1, dx_dims);
    auto dout_stride_0 = phi::funcs::ComputeStride(0, dout_dims);
    dx_data += (offset >= 0 ? offset * dx_stride_1 : -offset * dx_stride_0);

    auto dout_length = dout_dims[0];
    for (int i = 0; i < dout_length; i++) {
      dx_data[i * (dx_stride_0 + dx_stride_1)] = dout_data[i * dout_stride_0];
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(diag_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::DiagGradKernel,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   float,
                   double) {}
