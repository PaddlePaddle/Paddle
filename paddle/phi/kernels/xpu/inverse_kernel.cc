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

#include "paddle/phi/kernels/inverse_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void InverseKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);

  const auto& mat_dims = x.dims();
  const int rank = mat_dims.size();
  int n = mat_dims[rank - 1];
  int batch_size = rank > 2 ? x.numel() / (n * n) : 1;

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int* info_out = RAII_GUARD.alloc_l3_or_gm<int>(batch_size);
  PADDLE_ENFORCE_XDNN_NOT_NULL(info_out);

  int r = xpu::inverse<XPUType>(dev_ctx.x_context(),
                                reinterpret_cast<const XPUType*>(x.data<T>()),
                                reinterpret_cast<XPUType*>(out->data<T>()),
                                info_out,
                                batch_size,
                                n);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "inverse");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    inverse, XPU, ALL_LAYOUT, phi::InverseKernel, float, double) {}
