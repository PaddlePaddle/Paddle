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

#include "paddle/phi/kernels/inverse_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void InverseKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto out_data = dev_ctx.template Alloc<T>(out);

  int64_t x_dims_len = x.dims().size();
  PADDLE_ENFORCE_GT(
      x_dims_len,
      1,
      common::errors::InvalidArgument(
          "Dimensions of input should be greater than 1, but got %d.",
          x_dims_len));

  int64_t n = x.dims()[x_dims_len - 1];
  int64_t batch = x_dims_len > 2 ? x.numel() / (n * n) : 1;
  PADDLE_ENFORCE_LE(n * n * sizeof(T),
                    8192,
                    common::errors::InvalidArgument(
                        "The size of a single matrix (%d bytes) exceeds the "
                        "maximum numbers of bytes xpu supports (8192).",
                        n * n * sizeof(T)));
  auto RAII_GUARD = xpu::ctx_guard(dev_ctx.x_context());
  auto* info_xpu = RAII_GUARD.alloc_l3_or_gm<int>(batch);
  // Xpu inverse api has check for singularity itself.
  int r = xpu::inverse<XPUType>(dev_ctx.x_context(),
                                reinterpret_cast<const XPUType*>(x.data<T>()),
                                reinterpret_cast<XPUType*>(out_data),
                                info_xpu,
                                batch,
                                n);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "inverse");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    inverse, XPU, ALL_LAYOUT, phi::InverseKernel, float, double) {}
