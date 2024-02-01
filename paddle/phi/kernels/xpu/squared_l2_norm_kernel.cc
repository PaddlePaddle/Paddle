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

#include "paddle/phi/kernels/squared_l2_norm_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SquaredL2NormKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         DenseTensor* out) {
  T* data = dev_ctx.template Alloc<T>(out);
  using XPUType = typename XPUTypeTrait<T>::Type;

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  float* y_for_xdnn = nullptr;
  if (std::is_same<T, float>::value) {
    y_for_xdnn = reinterpret_cast<float*>(data);
  } else {
    y_for_xdnn = RAII_GUARD.alloc_l3_or_gm<float>(1);
  }

  // int square_reduce_sum(Context* ctx, const T* x, float* y, int64_t len, bool
  // is_sqrt=false);
  int r = xpu::square_reduce_sum<XPUType>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x.data<T>()),
      y_for_xdnn,
      x.numel(),
      false);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "square_reduce_sum");

  if (!std::is_same<T, float>::value) {
    // int cast(Context* ctx, const TX* x, TY* y, int64_t len);
    int r = xpu::cast<float, XPUType>(
        dev_ctx.x_context(), y_for_xdnn, reinterpret_cast<XPUType*>(data), 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(squared_l2_norm,
                   XPU,
                   ALL_LAYOUT,
                   phi::SquaredL2NormKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
