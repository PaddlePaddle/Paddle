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

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

namespace phi {

template <typename T, typename Context>
void MaximumWithAxisKernel(const Context& dev_ctx,
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
    return xpu::broadcast_max<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void MinimumWithAxisKernel(const Context& dev_ctx,
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
    return xpu::broadcast_min<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void RemainderWithAxisKernel(const Context& dev_ctx,
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
    return xpu::broadcast_mod<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void FloorDivideWithAxisKernel(const Context& dev_ctx,
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
    return xpu::broadcast_floordiv<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void ElementwisePowWithAxisKernel(const Context& dev_ctx,
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
    return xpu::broadcast_pow<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

}  // namespace phi

PD_REGISTER_KERNEL(floor_divide_with_axis,
                   XPU,
                   ALL_LAYOUT,
                   phi::FloorDivideWithAxisKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(maximum_with_axis,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaximumWithAxisKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(minimum_with_axis,
                   XPU,
                   ALL_LAYOUT,
                   phi::MinimumWithAxisKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(remainder_with_axis,
                   XPU,
                   ALL_LAYOUT,
                   phi::RemainderWithAxisKernel,
                   float,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow_with_axis,
                   XPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowWithAxisKernel,
                   float,
                   phi::dtype::float16) {}
