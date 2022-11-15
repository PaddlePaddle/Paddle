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

#include "paddle/phi/kernels/eig_grad_kernel.h"
#include "paddle/phi/kernels/cpu/eig.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void EigGradKernel(const Context& dev_ctx,
                   const DenseTensor& out_w,
                   const DenseTensor& out_v,
                   const DenseTensor& dout_w,
                   const DenseTensor& dout_v,
                   DenseTensor* dx) {
  auto* dx_data = dev_ctx.template Alloc<phi::dtype::Complex<T>>(dx);

  auto& dims = out_v.dims();
  phi::DDim dim_origin = dims;
  int num_dims = dim_origin.size();
  int batch_count = BatchCount(out_v);
  const int order = dim_origin[num_dims - 1];

  ComputeBackwardForComplexInput<phi::dtype::Complex<T>, Context>(
      out_w, out_v, dout_w, dout_v, dx_data, batch_count, order, dev_ctx);
}

}  // namespace phi

PD_REGISTER_KERNEL(eig_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::EigGradKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
  kernel->InputAt(2).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
