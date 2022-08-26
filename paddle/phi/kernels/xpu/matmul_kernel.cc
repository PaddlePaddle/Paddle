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

#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  dev_ctx.template Alloc<T>(out);
  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y.data<T>());
  XPUType* out_ptr = reinterpret_cast<XPUType*>(out->data<T>());
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  XpuFcInfo fc_info;
  GetFCInfo(x_dims, y_dims, trans_x, trans_y, &fc_info);
  xpu::Context* xpu_ctx = dev_ctx.x_context();
  MatMulXPUFunction<XPUType>(xpu_ctx, x_ptr, y_ptr, out_ptr, fc_info, 1.0f);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    matmul, XPU, ALL_LAYOUT, phi::MatmulKernel, float, phi::float16) {}
