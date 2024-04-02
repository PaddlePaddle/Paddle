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

#include "paddle/phi/kernels/elementwise_add_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

namespace phi {

template <typename T, typename Context>
void AddRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  int axis,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int>& xshape,
              const std::vector<int>& yshape) {
    return xpu::broadcast_add<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

}  // namespace phi

PD_REGISTER_KERNEL(add_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::AddRawKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   int,
                   int64_t) {}
