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
#include "xpu/refactor/paddle_api.h"

namespace phi {

template <typename T, typename Context>
void MaximumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* xpu_ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_max<XPUType>(xpu_ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void MinimumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* xpu_ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_min<XPUType>(xpu_ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void RemainderRawKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        int axis,
                        DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* xpu_ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_mod<XPUType>(xpu_ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void FloorDivideRawKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          int axis,
                          DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* xpu_ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_floordiv<XPUType>(xpu_ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void ElementwisePowRawKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int axis,
                             DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* xpu_ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_pow<XPUType>(xpu_ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

// For int64_t pow_raw, cast to float for computation then round back to int64.
template <>
void ElementwisePowRawKernel<int64_t, XPUContext>(const XPUContext& dev_ctx,
                                                  const DenseTensor& x,
                                                  const DenseTensor& y,
                                                  int axis,
                                                  DenseTensor* out) {
  dev_ctx.template Alloc<int64_t>(out);
  if (out->numel() == 0) return;

  DenseTensor x_float, y_float;
  x_float.Resize(x.dims());
  y_float.Resize(y.dims());
  dev_ctx.template Alloc<float>(&x_float);
  dev_ctx.template Alloc<float>(&y_float);

  int r = xpu::cast<int64_t, float>(
      dev_ctx.x_context(), x.data<int64_t>(), x_float.data<float>(), x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast int64 to float");

  r = xpu::cast<int64_t, float>(
      dev_ctx.x_context(), y.data<int64_t>(), y_float.data<float>(), y.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast int64 to float");

  DenseTensor out_float;
  out_float.Resize(out->dims());
  dev_ctx.template Alloc<float>(&out_float);

  auto f = [](xpu::Context* xpu_ctx,
              const float* x,
              const float* y,
              float* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_pow<float>(xpu_ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<float, float>(dev_ctx, x_float, y_float, axis, &out_float, f);

  r = xpu::paddle_round<float>(dev_ctx.x_context(),
                               out_float.data<float>(),
                               out_float.data<float>(),
                               out_float.numel(),
                               0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_round");

  r = xpu::cast<float, int64_t>(dev_ctx.x_context(),
                                out_float.data<float>(),
                                out->data<int64_t>(),
                                out_float.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast float to int64");
}

}  // namespace phi

PD_REGISTER_KERNEL(floor_divide_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::FloorDivideRawKernel,
                   float,
                   phi::bfloat16,
                   phi::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(maximum_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaximumRawKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(minimum_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::MinimumRawKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(remainder_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::RemainderRawKernel,
                   float,
                   phi::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowRawKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int64_t) {}
