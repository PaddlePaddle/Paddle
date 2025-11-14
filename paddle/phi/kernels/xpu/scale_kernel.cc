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

#include "paddle/phi/kernels/scale_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"

namespace phi {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 const Scalar& bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_EQ(
      x.dims(),
      out->dims(),
      common::errors::InvalidArgument("In and out should have the same dim,"
                                      " expected %s, but got %s.",
                                      x.dims().to_str().c_str(),
                                      out->dims().to_str().c_str()));
  if (x.numel() == 0 || !x.IsInitialized()) {
    return;
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  int r = xpu::scale<XPUType, MT>(dev_ctx.x_context(),
                                  reinterpret_cast<const XPUType*>(x.data<T>()),
                                  reinterpret_cast<XPUType*>(out->data<T>()),
                                  x.numel(),
                                  bias_after_scale,
                                  scale.to<MT>(),
                                  bias.to<MT>());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
}

template <>
void ScaleKernel<double, XPUContext>(const XPUContext& dev_ctx,
                                     const DenseTensor& x,
                                     const Scalar& scale,
                                     const Scalar& bias,
                                     bool bias_after_scale,
                                     DenseTensor* out) {
  auto x_float = phi::Cast<double>(dev_ctx, x, phi::DataType::FLOAT32);
  DenseTensor out_float;
  out_float.Resize(out->dims());
  ScaleKernel<float>(dev_ctx, x_float, scale, bias, bias_after_scale, &out_float);
  CastKernel<float>(dev_ctx, out_float, phi::DataType::FLOAT64, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(scale,
                   XPU,
                   ALL_LAYOUT,
                   phi::ScaleKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
