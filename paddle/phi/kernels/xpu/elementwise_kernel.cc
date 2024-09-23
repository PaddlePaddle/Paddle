//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_kernel.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  int axis = -1;
  FloorDivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MinimumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void RemainderKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int>& xshape,
              const std::vector<int>& yshape) {
    return xpu::broadcast_mod<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, -1, out, f);
}

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  int axis = -1;
  ElementwisePowRawKernel<T>(dev_ctx, x, y, axis, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(floor_divide,
                   XPU,
                   ALL_LAYOUT,
                   phi::FloorDivideKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(maximum,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaximumKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(minimum,
                   XPU,
                   ALL_LAYOUT,
                   phi::MinimumKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(remainder,
                   XPU,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow,
                   XPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
