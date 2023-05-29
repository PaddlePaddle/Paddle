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

#include "paddle/phi/kernels/momentum_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MomentumDenseKernel(const Context& dev_ctx,
                         const DenseTensor& param,
                         const DenseTensor& grad,
                         const DenseTensor& velocity,
                         const DenseTensor& learning_rate,
                         const paddle::optional<DenseTensor>& master_param,
                         float mu,
                         bool use_nesterov,
                         const std::string& regularization_method,
                         float regularization_coeff,
                         bool multi_precision,
                         float rescale_grad,
                         DenseTensor* param_out,
                         DenseTensor* velocity_out,
                         DenseTensor* master_param_out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(velocity_out);
  auto* lr = learning_rate.data<float>();

  if (regularization_method != "l2_decay") {
    // only support l2_decay
    regularization_coeff = 0.0f;
  }

  // int momentum(Context* ctx, const T* param, const T* velocity, const T*
  // grad, T* param_out, T* velocity_out, int len, const float* lr, int
  // use_nesterov, float mu, float l2_weight_decay);
  int r = xpu::momentum(dev_ctx.x_context(),
                        reinterpret_cast<const XPUType*>(param.data<T>()),
                        reinterpret_cast<const XPUType*>(velocity.data<T>()),
                        reinterpret_cast<const XPUType*>(grad.data<T>()),
                        reinterpret_cast<XPUType*>(param_out->data<T>()),
                        reinterpret_cast<XPUType*>(velocity_out->data<T>()),
                        param_out->numel(),
                        lr,
                        use_nesterov,
                        static_cast<T>(mu),
                        regularization_coeff);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "momentum");
}
}  // namespace phi

PD_REGISTER_KERNEL(momentum,
                   XPU,
                   ALL_LAYOUT,
                   phi::MomentumDenseKernel,
                   float,
                   phi::dtype::float16) {}
