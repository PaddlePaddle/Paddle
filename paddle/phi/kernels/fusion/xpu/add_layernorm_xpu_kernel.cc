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
void AddLayernormXPUKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           const DenseTensor& scale,
                           const DenseTensor& bias,
                           int64_t m,
                           int64_t n,
                           float epsilon,
                           DenseTensor* out,
                           DenseTensor* mean,
                           DenseTensor* variance,
                           DenseTensor* z_add) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  const float* scale_data = scale.data<float>();
  const float* bias_data = bias.data<float>();

  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  float* mean_data = ctx.template Alloc<float>(mean);
  float* variance_data = ctx.template Alloc<float>(variance);
  auto* z_add_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(z_add));

  int r = xpu::add_layer_norm_fusion<XPUType>(  // T
      /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
      /* const T* x */ x_data,
      /* const T* y */ y_data,
      /* T* z */ out_data,
      /* int64_t m */ m,
      /* int64_t n */ n,
      /* float epsilon */ epsilon,
      /* const float* scale */ scale_data,
      /* const float* bias */ bias_data,
      /* float* mean */ mean_data,
      /* float* variance */ variance_data,
      /* T* z_add */ z_add_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "add_layernorm_xpu");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(add_layernorm_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::AddLayernormXPUKernel,
                   float,
                   phi::dtype::float16) {}
