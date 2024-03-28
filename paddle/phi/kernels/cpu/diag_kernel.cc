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

#include "paddle/phi/kernels/diag_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/diag_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void DiagKernel(const Context& dev_ctx,
                const DenseTensor& x,
                int offset,
                float padding_value,
                DenseTensor* out) {
  auto* x_data = x.data<T>();
  auto x_dims = x.dims();
  T* out_data = dev_ctx.template Alloc<T>(out);
  auto out_dims = out->dims();

  int64_t i = 0;
  if (x_dims.size() <= 1) {
    phi::funcs::SetConstant<Context, T> set_padding_value;
    set_padding_value(dev_ctx, out, static_cast<T>(padding_value));

    auto x_length = (x_dims.size() == 1UL ? x_dims[0] : int64_t(1));
    const int& x_stride = 1;

    auto out_stride_0 = phi::funcs::ComputeStride(0, out_dims);
    auto out_stride_1 = phi::funcs::ComputeStride(1, out_dims);
    out_data += (offset >= 0 ? offset * out_stride_1 : -offset * out_stride_0);

    for (i = 0; i < x_length; i++) {
      out_data[i * (out_stride_0 + out_stride_1)] = x_data[i * x_stride];
    }
  } else {
    auto out_length = out_dims[0];
    const int& x_stride_0 = phi::funcs::ComputeStride(0, x_dims);
    const int& x_stride_1 = phi::funcs::ComputeStride(1, x_dims);

    auto out_stride_0 = phi::funcs::ComputeStride(0, out_dims);
    x_data += (offset >= 0 ? offset * x_stride_1 : -offset * x_stride_0);
    for (i = 0; i < out_length; i++) {
      out_data[i * out_stride_0] = x_data[i * (x_stride_0 + x_stride_1)];
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(diag,
                   CPU,
                   ALL_LAYOUT,
                   phi::DiagKernel,
                   phi::dtype::float16,
                   int,
                   float,
                   double,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
