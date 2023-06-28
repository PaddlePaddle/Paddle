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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void EltMulMulAddXPUKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           const DenseTensor& z,
                           DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  // (1-z) * x + z * y = z * (y - x) + x, z[batch, 1], x/y [batch, n]
  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  auto* z_data = reinterpret_cast<const XPUType*>(z.data<T>());
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  std::vector<int64_t> x_shape = phi::vectorize(x.dims());
  std::vector<int64_t> z_shape = phi::vectorize(z.dims());
  // out = y - x
  int64_t len = x.numel();
  int r = xpu::sub<XPUType>(ctx.x_context(), y_data, x_data, out_data, len);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "elementwise_mul_mul_add_xpu");
  // out = out * z
  r = xpu::broadcast_mul<XPUType>(
      ctx.x_context(), out_data, z_data, out_data, x_shape, z_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "elementwise_mul_mul_add_xpu");
  // out = out + x
  r = xpu::add<XPUType>(ctx.x_context(), out_data, x_data, out_data, len);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "elementwise_mul_mul_add_xpu");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(elementwise_mul_mul_add_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::EltMulMulAddXPUKernel,
                   float,
                   phi::dtype::float16) {}
