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

#include "paddle/phi/kernels/elementwise_add_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

namespace phi {

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  if (x.dtype() == phi::DataType::FLOAT32 &&
      (y.dtype() == phi::DataType::BFLOAT16 ||
       y.dtype() == phi::DataType::FLOAT16)) {
    // special case for "float32 + bfloat16", or "float32 + float16"
    auto dev_version =
        phi::backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId());
    if (dev_version >= phi::backends::xpu::XPUVersion::XPU3 &&
        x.dims() == y.dims()) {
      dev_ctx.template Alloc<float>(out);

      const float* x_data = x.data<float>();
      float* z_data = out->data<float>();

      int ret = xpu::SUCCESS;
      if (y.dtype() == phi::DataType::BFLOAT16) {
        using YType = DataTypeToCppType<phi::DataType::BFLOAT16>::type;
        using XPUYType = typename XPUTypeTrait<YType>::Type;
        auto y_data = reinterpret_cast<const XPUYType*>(y.data<YType>());
        ret = xpu::add_mul_type<float, XPUYType, float>(
            dev_ctx.x_context(), x_data, y_data, z_data, x.numel());
      } else {
        using YType = DataTypeToCppType<phi::DataType::FLOAT16>::type;
        using XPUYType = typename XPUTypeTrait<YType>::Type;
        auto y_data = reinterpret_cast<const XPUYType*>(y.data<YType>());
        ret = xpu::add_mul_type<float, XPUYType, float>(
            dev_ctx.x_context(), x_data, y_data, z_data, x.numel());
      }
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "add_mul_type");
    } else {
      using Type = DataTypeToCppType<phi::DataType::FLOAT32>::type;
      using XPUType = typename XPUTypeTrait<Type>::Type;
      auto f = [](xpu::Context* ctx,
                  const XPUType* x,
                  const XPUType* y,
                  XPUType* z,
                  const std::vector<int>& xshape,
                  const std::vector<int>& yshape) {
        return xpu::broadcast_add<XPUType>(ctx, x, y, z, xshape, yshape);
      };
      auto casted_y = phi::Cast<T>(dev_ctx, y, phi::DataType::FLOAT32);
      XPUElementwise<Type, XPUType>(dev_ctx, x, casted_y, -1, out, f);
    }
  } else {
    using XPUType = typename XPUTypeTrait<T>::Type;

    auto f = [](xpu::Context* ctx,
                const XPUType* x,
                const XPUType* y,
                XPUType* z,
                const std::vector<int>& xshape,
                const std::vector<int>& yshape) {
      return xpu::broadcast_add<XPUType>(ctx, x, y, z, xshape, yshape);
    };

    XPUElementwise<T, XPUType>(dev_ctx, x, y, -1, out, f);
  }
}

template <typename T, typename Context>
void GradAddXPUKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  dev_ctx.template Alloc<T>(out);
  auto x_shape = common::vectorize<int>(x.dims());
  auto y_shape = common::vectorize<int>(y.dims());
  int r = xpu::broadcast_add(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x.data<T>()),
                             reinterpret_cast<const XPUType*>(y.data<T>()),
                             reinterpret_cast<XPUType*>(out->data<T>()),
                             x_shape,
                             y_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
}

}  // namespace phi

PD_REGISTER_KERNEL(grad_add,
                   XPU,
                   ALL_LAYOUT,
                   phi::GradAddXPUKernel,
                   phi::dtype::float16,
                   float) {}

PD_REGISTER_KERNEL(add,
                   XPU,
                   ALL_LAYOUT,
                   phi::AddKernel,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   int,
                   int64_t) {}
