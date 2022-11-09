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

#include "paddle/phi/kernels/adamw_kernel.h"

#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
// for TensorToVector
#include "paddle/fluid/framework/tensor_util.h"

namespace phi {

template <typename T, typename Context>
void AdamwDenseKernel(const Context& dev_ctx,
                      const DenseTensor& param,
                      const DenseTensor& grad,
                      const DenseTensor& learning_rate,
                      const DenseTensor& moment1,
                      const DenseTensor& moment2,
                      const DenseTensor& beta1_pow,
                      const DenseTensor& beta2_pow,
                      const paddle::optional<DenseTensor>& master_param,
                      const paddle::optional<DenseTensor>& skip_update,
                      const Scalar& beta1,
                      const Scalar& beta2,
                      const Scalar& epsilon,
                      float lr_ratio,
                      float coeff,
                      bool with_decay,
                      bool lazy_mode,
                      int64_t min_row_size_to_use_multithread,
                      bool multi_precision,
                      bool use_global_beta_pow,
                      DenseTensor* param_out,
                      DenseTensor* moment1_out,
                      DenseTensor* moment2_out,
                      DenseTensor* beta1_pow_out,
                      DenseTensor* beta2_pow_out,
                      DenseTensor* master_param_outs) {
  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    std::vector<bool> skip_update_vec;
    paddle::framework::TensorToVector(*skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }
  if (skip_update_) {
    VLOG(4) << "Adamw skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
    phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    return;
  }

  auto beta1_ = beta1.to<float>();
  auto beta2_ = beta2.to<float>();
  auto epsilon_ = epsilon.to<float>();

  const float* beta1_pow_ptr = beta1_pow.template data<float>();
  const float* beta2_pow_ptr = beta2_pow.template data<float>();
  DenseTensor xpu_beta1_pow;
  DenseTensor xpu_beta2_pow;
  if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
    phi::Copy(dev_ctx, beta1_pow, dev_ctx.GetPlace(), false, &xpu_beta1_pow);
    phi::Copy(dev_ctx, beta2_pow, dev_ctx.GetPlace(), false, &xpu_beta2_pow);
    dev_ctx.Wait();
    beta1_pow_ptr = xpu_beta1_pow.template data<float>();
    beta2_pow_ptr = xpu_beta2_pow.template data<float>();
  }
  if (with_decay) {
    int r = xpu::adamw(dev_ctx.x_context(),
                       grad.template data<T>(),
                       moment1.template data<float>(),
                       moment2.template data<float>(),
                       param.template data<T>(),
                       beta1_pow_ptr,
                       beta2_pow_ptr,
                       learning_rate.template data<float>(),
                       dev_ctx.template Alloc<float>(moment1_out),
                       dev_ctx.template Alloc<float>(moment2_out),
                       dev_ctx.template Alloc<T>(param_out),
                       beta1_,
                       beta2_,
                       epsilon_,
                       coeff,
                       param.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");
  } else {
    int r = xpu::adam(dev_ctx.x_context(),
                      grad.template data<T>(),
                      moment1.template data<float>(),
                      moment2.template data<float>(),
                      param.template data<T>(),
                      beta1_pow_ptr,
                      beta2_pow_ptr,
                      learning_rate.template data<float>(),
                      dev_ctx.template Alloc<float>(moment1_out),
                      dev_ctx.template Alloc<float>(moment2_out),
                      dev_ctx.template Alloc<T>(param_out),
                      beta1_,
                      beta2_,
                      epsilon_,
                      param.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");
  }

  if (!use_global_beta_pow) {
    // update in cpu and then copy to xpu
    if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
      const float* beta1_pow_p = beta1_pow.template data<float>();
      dev_ctx.template HostAlloc<float>(beta1_pow_out)[0] =
          beta1_ * beta1_pow_p[0];
      const float* beta2_pow_p = beta2_pow.template data<float>();
      dev_ctx.template HostAlloc<float>(beta2_pow_out)[0] =
          beta2_ * beta2_pow_p[0];
      xpu_wait(dev_ctx.x_context()->xpu_stream);
    } else {
      float* beta1_pow_out_p = dev_ctx.template Alloc<float>(beta1_pow_out);
      float* beta2_pow_out_p = dev_ctx.template Alloc<float>(beta2_pow_out);
      int r = xpu::scale(dev_ctx.x_context(),
                         beta1_pow_ptr,
                         beta1_pow_out_p,
                         beta1_pow.numel(),
                         false,
                         beta1_,
                         0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");
      r = xpu::scale(dev_ctx.x_context(),
                     beta2_pow_ptr,
                     beta2_pow_out_p,
                     beta2_pow.numel(),
                     false,
                     beta2_,
                     0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(adamw, XPU, ALL_LAYOUT, phi::AdamwDenseKernel, float) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
}
