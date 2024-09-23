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

#include "paddle/phi/kernels/addmm_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void AddmmKernel(const Context& dev_ctx,
                 const DenseTensor& input,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 float beta,
                 float alpha,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  dev_ctx.template Alloc<T>(out);
  //phi::Copy(dev_ctx, input, dev_ctx.GetPlace(), false, out);
  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y.data<T>());
  XPUType* out_ptr = reinterpret_cast<XPUType*>(out->data<T>());

  auto input_dims = input.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(input_dims.size() == 2 || input_dims.size() == 1,
                    true,
                    common::errors::InvalidArgument(
                        "Variable 'input' of AddmmOp must be 1-dimensional or 2-dimensional, "
                        "but received shape: [%s]",
                        input_dims));
  PADDLE_ENFORCE_EQ(x_dims.size() == 2,
                    true,
                    common::errors::InvalidArgument(
                        "Variable 'x' of AddmmOp must be 2-dimensional, "
                        "but received shape: [%s]",
                        input_dims));
  PADDLE_ENFORCE_EQ(y_dims.size() == 2,
                    true,
                    common::errors::InvalidArgument(
                        "Variable 'y' of AddmmOp must be 2-dimensional, "
                        "but received shape: [%s]",
                        input_dims));

  XpuFcInfo fc_info;
  GetFCInfo(x_dims, y_dims, false, false, &fc_info);
  xpu::Context* xpu_ctx = dev_ctx.x_context();
  MatMulXPUFunction<XPUType>(xpu_ctx, x_ptr, y_ptr, out_ptr, fc_info, alpha, beta);

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  T* tmp = RAII_GUARD.alloc_l3_or_gm<T>(input.numel());
  int r = xpu::scale(dev_ctx.x_context(),
                     reinterpret_cast<const XPUType*>(input.data<T>()),
                     reinterpret_cast<XPUType*>(tmp),
                     input.numel(),
                     true,
                     beta,
                     0.f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  auto out_dims = out->dims();
  auto out_dims_vec = common::vectorize<int64_t>(out_dims);
  auto input_dims_vec = common::vectorize<int64_t>(input_dims);
  r = xpu::broadcast_add(
    dev_ctx.x_context(),
    out_ptr,
    reinterpret_cast<XPUType*>(tmp),
    out_ptr,
    out_dims_vec,
    input_dims_vec);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
}
}  // namespace phi

PD_REGISTER_KERNEL(addmm,
                   XPU,
                   ALL_LAYOUT,
                   phi::AddmmKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
