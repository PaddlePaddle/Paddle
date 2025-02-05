// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/expand_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ExpandGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const IntArray& shape,
                      DenseTensor* in_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto in_grad_data = ctx.template Alloc<T>(in_grad);
  auto out_grad_dims = common::vectorize<int64_t>(out_grad.dims());
  auto in_grad_dims = common::vectorize<int64_t>(in_grad->dims());
  in_grad_dims.insert(
      in_grad_dims.begin(), out_grad.dims().size() - in_grad->dims().size(), 1);

  // Two zero
  if (out_grad_dims.size() == 0 && in_grad_dims.size() == 0) {
    out_grad_dims = {1};
    in_grad_dims = {1};
  }

  int r = xpu::expand_grad<XPUType>(
      ctx.x_context(),
      reinterpret_cast<const XPUType*>(out_grad.data<T>()),
      reinterpret_cast<XPUType*>(in_grad_data),
      out_grad_dims,
      in_grad_dims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "expand_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(expand_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ExpandGradKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
