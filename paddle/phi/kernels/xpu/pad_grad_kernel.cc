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

#include "paddle/phi/kernels/pad_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void PadGradKernel(const Context& dev_ctx,
                   const DenseTensor& d_out,
                   const std::vector<int>& paddings,
                   const Scalar& pad_value,
                   DenseTensor* d_x) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  std::vector<int> pad_left, pad_right;
  std::vector<int> out_shape = common::vectorize<int>(d_out.dims());
  dev_ctx.template Alloc<T>(d_x);

  for (size_t i = 0; i < paddings.size() / 2; ++i) {
    pad_left.push_back(-paddings[i * 2]);
    pad_right.push_back(-paddings[i * 2 + 1]);
  }

  XPUType value = static_cast<XPUType>(pad_value.to<T>());
  int r = xpu::pad<XPUType>(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(d_out.data<T>()),
                            reinterpret_cast<XPUType*>(d_x->data<T>()),
                            out_shape,
                            pad_left,
                            pad_right,
                            value);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
}
}  // namespace phi

PD_REGISTER_KERNEL(pad_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::PadGradKernel,
                   float,
                   int,
                   int16_t,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
