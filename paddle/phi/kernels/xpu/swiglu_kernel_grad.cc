// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/swiglu_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void SwiGluGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& y,
                      const DenseTensor& dz,
                      DenseTensor* dx,
                      DenseTensor* dy) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto* x_data = x.data<T>();
  const auto* dz_data = dz.data<T>();
  auto* dx_data = ctx.template Alloc<T>(dx);
  const auto& dims = x.dims();
  int64_t axis = dims.size() - 1;
  auto dims_vec = common::vectorize<int64_t>(dims);
  const XPUType* y_ptr = nullptr;
  XPUType* dy_ptr = nullptr;

  if (y) {
    const auto& y_tensor = y.get();
    const auto& y_dims = y_tensor.dims();
    const auto* y_data = y_tensor.data<T>();
    auto* dy_data = ctx.template Alloc<T>(dy);
    y_ptr = reinterpret_cast<const XPUType*>(y_data);
    dy_ptr = reinterpret_cast<XPUType*>(dy_data);
    PADDLE_ENFORCE_EQ(y_dims,
                      dims,
                      common::errors::InvalidArgument(
                          "The shape of Input(Y):[%s] must be equal "
                          "to the shape of Input(X):[%s].",
                          y_dims,
                          dims));
  }
  int ret = xpu::swiglu_grad(ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x_data),
                             reinterpret_cast<const XPUType*>(dz_data),
                             reinterpret_cast<XPUType*>(dx_data),
                             dims_vec,
                             axis,
                             true,
                             y_ptr,
                             dy_ptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "swiglu_grad");
}
}  // namespace phi
PD_REGISTER_KERNEL(swiglu_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SwiGluGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
