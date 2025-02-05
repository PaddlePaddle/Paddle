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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename TX, typename TY, typename Context>
void DeQuantizeKernelImpl(const Context& ctx,
                          const DenseTensor& x,
                          float scale,
                          DenseTensor* y) {
  using XPUInX = typename XPUTypeTrait<TX>::Type;
  using XPUOutY = typename XPUTypeTrait<TY>::Type;

  auto* y_data = ctx.template Alloc<TY>(y);
  const auto* x_data = x.data<TX>();
  int64_t len = x.numel();
  int max_ptr_size = ctx.x_context()->max_ptr_size();
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  auto max_data = RAII_GUARD.alloc_l3_or_gm<float>(max_ptr_size);
  int r = xpu::constant<float>(ctx.x_context(), max_data, max_ptr_size, scale);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  r = xpu::dequantization<XPUInX, XPUOutY>(
      ctx.x_context(),
      reinterpret_cast<const XPUInX*>(x_data),
      reinterpret_cast<XPUOutY*>(y_data),
      len,
      max_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "dequantization");
}

template <typename T, typename Context>
void DeQuantizeKernel(const Context& ctx,
                      const DenseTensor& x,
                      DataType out_dtype,
                      float scale,
                      DenseTensor* y) {
  switch (out_dtype) {
    case DataType::FLOAT32:
      DeQuantizeKernelImpl<T, float, Context>(ctx, x, scale, y);
      break;
    case DataType::FLOAT16:
      DeQuantizeKernelImpl<T, dtype::float16, Context>(ctx, x, scale, y);
      break;
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "Not supported dequantize data type from %d -> %d ",
          x.dtype(),
          out_dtype));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    dequantize_xpu, XPU, ALL_LAYOUT, phi::DeQuantizeKernel, int16_t, int8_t) {}
