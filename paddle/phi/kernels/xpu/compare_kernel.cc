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

<<<<<<< HEAD
  if (x.dims().size() == 0) {
    x_shape = std::vector<int>({1});
  }
  if (y.dims().size() == 0) {
    y_shape = std::vector<int>({1});
  }

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  auto x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  auto* out_data = dev_ctx.template Alloc<bool>(out);

  int ret =
      func(dev_ctx.x_context(), x_data, y_data, out_data, x_shape, y_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "compare op");
}

<<<<<<< HEAD
#define DEFINE_XPU_COMPARE_KERNEL(name, functor)                      \
  template <typename T, typename Context>                             \
  void name##RawKernel(const Context& dev_ctx,                        \
                       const DenseTensor& x,                          \
                       const DenseTensor& y,                          \
                       int axis,                                      \
                       DenseTensor* out) {                            \
    using XPUType = typename XPUTypeTrait<T>::Type;                   \
    auto f = [](xpu::Context* ctx,                                    \
                const XPUType* x,                                     \
                const XPUType* y,                                     \
                bool* z,                                              \
                const std::vector<int>& xshape,                       \
                const std::vector<int>& yshape) {                     \
      return functor(ctx, x, y, z, xshape, yshape);                   \
    };                                                                \
    XPUCompareKernelImpl<T, XPUType, Context>(dev_ctx, x, y, out, f); \
  }                                                                   \
  template <typename T, typename Context>                             \
  void name##Kernel(const Context& dev_ctx,                           \
                    const DenseTensor& x,                             \
                    const DenseTensor& y,                             \
                    DenseTensor* out) {                               \
    name##RawKernel<T, Context>(dev_ctx, x, y, -1, out);              \
  }

DEFINE_XPU_COMPARE_KERNEL(Equal, xpu::broadcast_equal<XPUType>)
DEFINE_XPU_COMPARE_KERNEL(NotEqual, xpu::broadcast_not_equal<XPUType>)
DEFINE_XPU_COMPARE_KERNEL(LessThan, xpu::broadcast_less_than<XPUType>)
DEFINE_XPU_COMPARE_KERNEL(LessEqual, xpu::broadcast_less_equal<XPUType>)
DEFINE_XPU_COMPARE_KERNEL(GreaterThan, xpu::broadcast_greater_than<XPUType>)
DEFINE_XPU_COMPARE_KERNEL(GreaterEqual, xpu::broadcast_greater_equal<XPUType>)

=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#undef DEFINE_XPU_COMPARE_KERNEL

}  // namespace phi

PD_REGISTER_KERNEL(
<<<<<<< HEAD
    less_than, XPU, ALL_LAYOUT, phi::LessThanKernel, int, int64_t, float) {}

PD_REGISTER_KERNEL(less_than_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::LessThanRawKernel,
                   int,
                   int64_t,
                   float) {}

#define PD_REGISTER_COMPARE_KERNEL(name, func)                          \
  PD_REGISTER_KERNEL(                                                   \
      name, XPU, ALL_LAYOUT, phi::func##Kernel, int, int64_t, float) {} \
  PD_REGISTER_KERNEL(name##_raw,                                        \
                     XPU,                                               \
                     ALL_LAYOUT,                                        \
                     phi::func##RawKernel,                              \
                     int,                                               \
                     int64_t,                                           \
                     float) {}

PD_REGISTER_COMPARE_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_KERNEL(greater_equal, GreaterEqual)
PD_REGISTER_COMPARE_KERNEL(equal, Equal)
PD_REGISTER_COMPARE_KERNEL(not_equal, NotEqual)
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
