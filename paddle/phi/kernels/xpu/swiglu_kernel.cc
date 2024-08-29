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

#include "paddle/phi/kernels/swiglu_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void SwiGluKernel(const Context& ctx,
                  const DenseTensor& x,
                  const paddle::optional<DenseTensor>& y,
                  DenseTensor* z) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUTypefp32 = typename XPUTypeTrait<float>::Type;
  const auto* x_data = x.data<T>();
  auto* z_data = ctx.template Alloc<T>(z);
  const auto& dims = x.dims();
  int64_t axis = dims.size() - 1;
  auto dims_vec = common::vectorize<int64_t>(dims);
  const XPUTypefp32* const_nullptr = nullptr;
  const XPUType* y_ptr = nullptr;

  if (y) {
    const auto& y_tensor = y.get();
    const auto& y_dims = y_tensor.dims();
    const auto* y_data = y_tensor.data<T>();
    y_ptr = reinterpret_cast<const XPUType*>(y_data);
    PADDLE_ENFORCE_EQ(y_dims,
                      dims,
                      common::errors::InvalidArgument(
                          "The shape of Input(Y):[%s] must be equal "
                          "to the shape of Input(X):[%s].",
                          y_dims,
                          dims));
  }
  int ret = xpu::swiglu(ctx.x_context(),
                        reinterpret_cast<const XPUType*>(x_data),
                        reinterpret_cast<XPUType*>(z_data),
                        dims_vec,
                        axis,
                        true,
                        const_nullptr,
                        nullptr,
                        y_ptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "swiglu");
}
}  // namespace phi
PD_REGISTER_KERNEL(swiglu,
                   XPU,
                   ALL_LAYOUT,
                   phi::SwiGluKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16){};
