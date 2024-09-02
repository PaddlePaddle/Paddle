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

#include "paddle/phi/kernels/eye_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void EyeKernel(const Context& ctx,
               const Scalar& num_rows,
               const Scalar& num_columns,
               DataType dtype,
               DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(out);
  auto out_data = reinterpret_cast<XPUType*>(out->data<T>());
  int64_t num_rows_data = num_rows.to<int64_t>();
  int64_t num_columns_data = num_columns.to<int64_t>();

  int r = xpu::eye<XPUType>(
      ctx.x_context(), out_data, num_rows_data, num_columns_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "eye");
}

}  // namespace phi

PD_REGISTER_KERNEL(eye,
                   XPU,
                   ALL_LAYOUT,
                   phi::EyeKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
