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

#include "paddle/phi/kernels/masked_select_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& mask,
                            const DenseTensor& out_grad,
                            DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* mask_data = mask.data<bool>();
  auto* input_data = reinterpret_cast<const XPUType*>(out_grad.data<T>());
  auto* out_data =
      reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(x_grad));

  auto mask_shape = common::vectorize<int>(mask.dims());
  auto xshape = common::vectorize<int>(x_grad->dims());
  if (mask.dims().size() == 0) {
    mask_shape = std::vector<int>({1});
  }
  if (x_grad->dims().size() == 0) {
    xshape = std::vector<int>({1});
  }

  int r = xpu::masked_select_grad(dev_ctx.x_context(),
                                  input_data,
                                  mask_data,
                                  out_data,
                                  xshape,
                                  mask_shape,
                                  1);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "masked_select_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_select_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaskedSelectGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   bool,
                   int64_t) {}
