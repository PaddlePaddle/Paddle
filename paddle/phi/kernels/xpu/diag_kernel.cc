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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DiagKernel(const Context& dev_ctx,
                const DenseTensor& x,
                int offset,
                float padding_value,
                DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  dev_ctx.template Alloc<T>(out);
  auto* out_data = reinterpret_cast<XPUType*>(out->data<T>());

  auto x_shape = common::vectorize<int>(x.dims());
  auto out_shape = common::vectorize<int>(out->dims());

  if (x.dims().size() == 0) {
    x_shape = std::vector<int>({1});
  }

  int r = xpu::diag<XPUType>(dev_ctx.x_context(),
                             x_data,
                             out_data,
                             x_shape,
                             out_shape,
                             offset,
                             padding_value);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "diag");
}

}  // namespace phi

PD_REGISTER_KERNEL(diag,
                   XPU,
                   ALL_LAYOUT,
                   phi::DiagKernel,
                   phi::dtype::float16,
                   int,
                   float,
                   int64_t) {}
