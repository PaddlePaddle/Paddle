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

#include "paddle/phi/kernels/cumprod_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/cumprod.h"

namespace phi {
template <typename T, typename Context>
void CumprodKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   int dim,
                   bool exclusive,
                   bool reverse,
                   DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const DenseTensor* x = &input;
  auto* x_data = x->data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);
  DDim shape = x->dims();
  std::vector<int64_t> xshape = common::vectorize<int64_t>(shape);

  if (dim < 0) dim += xshape.size();
  if (shape.size() == 0) {
    int r =
        xpu::copy<XPUType>(dev_ctx.x_context(),
                           reinterpret_cast<const XPUType*>(input.data<T>()),
                           reinterpret_cast<XPUType*>(out->data<T>()),
                           input.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

    return;
  }

  int r = xpu::cumprod(dev_ctx.x_context(),
                       reinterpret_cast<const XPUType*>(x_data),
                       reinterpret_cast<XPUType*>(out_data),
                       xshape,
                       dim);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cumprod");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    cumprod, XPU, ALL_LAYOUT, phi::CumprodKernel, float, int, int64_t) {}
