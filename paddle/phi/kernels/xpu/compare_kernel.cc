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

#include "paddle/phi/kernels/compare_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename XPUType, typename Context>
void XPUCompareKernelImpl(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out,
                          std::function<int(xpu::Context*,
                                            const XPUType*,
                                            const XPUType*,
                                            bool*,
                                            const std::vector<int>&,
                                            const std::vector<int>&)> func) {
  auto x_shape = vectorize<int>(x.dims());
  auto y_shape = vectorize<int>(y.dims());

  auto x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  auto* out_data = dev_ctx.template Alloc<bool>(out);

  int ret =
      func(dev_ctx.x_context(), x_data, y_data, out_data, x_shape, y_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "compare op");
}

#define DEFINE_XPU_COMPARE_KERNEL(compare_kernel, functor)                  \
  template <typename T, typename Context>                                   \
  void compare_kernel(const Context& dev_ctx,                               \
                      const DenseTensor& x,                                 \
                      const DenseTensor& y,                                 \
                      int axis,                                             \
                      DenseTensor* out) {                                   \
    using XPUType = typename XPUTypeTrait<T>::Type;                         \
    XPUCompareKernelImpl<T, XPUType, Context>(dev_ctx, x, y, out, functor); \
  }

DEFINE_XPU_COMPARE_KERNEL(EqualKernel, xpu::broadcast_equal<XPUType>)
DEFINE_XPU_COMPARE_KERNEL(NotEqualKernel, xpu::broadcast_not_equal<XPUType>)
DEFINE_XPU_COMPARE_KERNEL(LessThanKernel, xpu::broadcast_less_than<XPUType>)
DEFINE_XPU_COMPARE_KERNEL(LessEqualKernel, xpu::broadcast_less_equal<XPUType>)
DEFINE_XPU_COMPARE_KERNEL(GreaterThanKernel,
                          xpu::broadcast_greater_than<XPUType>)
DEFINE_XPU_COMPARE_KERNEL(GreaterEqualKernel,
                          xpu::broadcast_greater_equal<XPUType>)
#undef DEFINE_XPU_COMPARE_KERNEL

}  // namespace phi

PD_REGISTER_KERNEL(
    equal, XPU, ALL_LAYOUT, phi::EqualKernel, float, int, int64_t) {}
PD_REGISTER_KERNEL(
    not_equal, XPU, ALL_LAYOUT, phi::NotEqualKernel, float, int, int64_t) {}
PD_REGISTER_KERNEL(
    less_than, XPU, ALL_LAYOUT, phi::LessThanKernel, float, int, int64_t) {}
PD_REGISTER_KERNEL(
    less_equal, XPU, ALL_LAYOUT, phi::LessEqualKernel, float, int, int64_t) {}
PD_REGISTER_KERNEL(greater_than,
                   XPU,
                   ALL_LAYOUT,
                   phi::GreaterThanKernel,
                   float,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(greater_equal,
                   XPU,
                   ALL_LAYOUT,
                   phi::GreaterEqualKernel,
                   float,
                   int,
                   int64_t) {}
