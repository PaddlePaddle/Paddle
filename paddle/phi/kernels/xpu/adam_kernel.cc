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

#include "paddle/phi/kernels/adam_kernel.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"

namespace phi {

using float16 = dtype::float16;

template <typename T, typename Context>
void AdamDenseKernel(const Context& dev_ctx,
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
  float* param_ptr = nullptr;
  funcs::GetDataPointer<Context, float>(param, &param_ptr, dev_ctx);

  float* mom1_ptr = nullptr;
  funcs::GetDataPointer<Context, float>(moment1, &mom1_ptr, dev_ctx);

  float* mom2_ptr = nullptr;
  funcs::GetDataPointer<Context, float>(moment2, &mom2_ptr, dev_ctx);

  float* lr_ptr = nullptr;
  funcs::GetDataPointer<Context, float>(learning_rate, &lr_ptr, dev_ctx);

  float* beta1_pow_ptr = nullptr;
  const float* beta1_const_pow_ptr = nullptr;
  if (beta1_pow.place() == CPUPlace()) {
    DenseTensor xpu_beta1_pow;
    phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, &xpu_beta1_pow);
    if (xpu_beta1_pow.dtype() == DataType::FLOAT16)
      funcs::GetDataPointer<Context, float>(
          xpu_beta1_pow, &beta1_pow_ptr, dev_ctx);
    else
      beta1_const_pow_ptr = xpu_beta1_pow.template data<float>();
  } else {
    if (beta1_pow.dtype() == DataType::FLOAT16)
      funcs::GetDataPointer<Context, float>(beta1_pow, &beta1_pow_ptr, dev_ctx);
    else
      beta1_const_pow_ptr = beta1_pow.template data<float>();
  }

  float* beta2_pow_ptr = nullptr;
  const float* beta2_const_pow_ptr = nullptr;
  if (beta2_pow.place() == CPUPlace()) {
    DenseTensor xpu_beta2_pow;
    phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, &xpu_beta2_pow);
    if (xpu_beta2_pow.dtype() == DataType::FLOAT16)
      funcs::GetDataPointer<Context, float>(
          xpu_beta2_pow, &beta2_pow_ptr, dev_ctx);
    else
      beta2_const_pow_ptr = xpu_beta2_pow.template data<float>();
  } else {
    if (beta2_pow.dtype() == DataType::FLOAT16)
      funcs::GetDataPointer<Context, float>(beta2_pow, &beta2_pow_ptr, dev_ctx);
    else
      beta2_const_pow_ptr = beta2_pow.template data<float>();
  }

  DenseTensor xpu_param_out;
  float* param_out_ptr = nullptr;
  const phi::DenseTensorMeta meta_param(DataType::FLOAT32, param_out->dims());
  xpu_param_out.set_meta(meta_param);
  funcs::GetOutDataPointer<Context, float>(
      param_out, &xpu_param_out, &param_out_ptr, dev_ctx);

  DenseTensor xpu_mom1_out;
  float* mom1_out_ptr = nullptr;
  const phi::DenseTensorMeta meta_mom1(DataType::FLOAT32, moment1_out->dims());
  xpu_mom1_out.set_meta(meta_mom1);
  funcs::GetOutDataPointer<Context, float>(
      moment1_out, &xpu_mom1_out, &mom1_out_ptr, dev_ctx);

  DenseTensor xpu_mom2_out;
  float* mom2_out_ptr = nullptr;
  const phi::DenseTensorMeta meta_mom2(DataType::FLOAT32, moment2_out->dims());
  xpu_mom2_out.set_meta(meta_mom2);
  funcs::GetOutDataPointer<Context, float>(
      moment2_out, &xpu_mom2_out, &mom2_out_ptr, dev_ctx);

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
    VLOG(4) << "Adam skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
    phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    return;
  }

  PADDLE_ENFORCE_EQ(
      beta1_pow_out->numel(),
      1,
      errors::InvalidArgument("Tensor holds the wrong size, Expected beta1 pow "
                              "output size is 1, but received "
                              "value is:%d.",
                              beta1_pow_out->numel()));

  PADDLE_ENFORCE_EQ(
      beta2_pow_out->numel(),
      1,
      errors::InvalidArgument("Tensor holds the wrong size, Expected beta2 pow "
                              "output size is 1, but received "
                              "value is:%d.",
                              beta2_pow_out->numel()));

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  auto beta1_ = beta1.to<float>();
  auto beta2_ = beta2.to<float>();
  auto epsilon_ = epsilon.to<float>();

  float* grad_c = nullptr;
  funcs::GetDataPointer<Context, float>(grad, &grad_c, dev_ctx);

  int r = xpu::adam(
      dev_ctx.x_context(),
      grad_c != nullptr ? grad_c : grad.template data<float>(),
      mom1_ptr != nullptr ? mom1_ptr : moment1.template data<float>(),
      mom2_ptr != nullptr ? mom2_ptr : moment2.template data<float>(),
      param_ptr != nullptr ? param_ptr : param.template data<float>(),
      beta1_pow_ptr != nullptr ? beta1_pow_ptr : beta1_const_pow_ptr,
      beta2_pow_ptr != nullptr ? beta2_pow_ptr : beta2_const_pow_ptr,
      lr_ptr != nullptr ? lr_ptr : learning_rate.template data<float>(),
      mom1_out_ptr,
      mom2_out_ptr,
      param_out_ptr,
      beta1_,
      beta2_,
      epsilon_,
      param.numel());

  xpu_wait(dev_ctx.x_context()->xpu_stream);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");

  funcs::FreeData<float>(grad, grad_c);

  funcs::CopyOutData<Context, float>(xpu_mom1_out, moment1_out, dev_ctx);
  funcs::CopyOutData<Context, float>(xpu_mom2_out, moment2_out, dev_ctx);
  funcs::CopyOutData<Context, float>(xpu_param_out, param_out, dev_ctx);

  if (!use_global_beta_pow) {
    // update in cpu and then copy to xpu
    if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
      funcs::SetBetaData<Context, float>(
          beta1_pow, beta1_pow_out, beta1_, dev_ctx);

      funcs::SetBetaData<Context, float>(
          beta2_pow, beta2_pow_out, beta2_, dev_ctx);
    } else {
      float* beta1_pow_out_p1 = nullptr;

      if (beta1_pow_out->dtype() == DataType::FLOAT16) {
        funcs::Scale<Context, float>(
            beta1_pow_out, beta1_pow, beta1_pow_ptr, beta1_, dev_ctx);
      } else {
        const float* beta1_pow_data = beta1_pow.template data<float>();
        beta1_pow_out_p1 = dev_ctx.template Alloc<float>(beta1_pow_out);
        r = xpu::scale(dev_ctx.x_context(),
                       beta1_pow_data,
                       beta1_pow_out_p1,
                       beta1_pow.numel(),
                       false,
                       beta1_,
                       0.0f);
        xpu_wait(dev_ctx.x_context()->xpu_stream);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");
      }

      float* beta2_pow_out_p1 = nullptr;
      if (beta2_pow_out->dtype() == DataType::FLOAT16) {
        funcs::Scale<Context, float>(
            beta2_pow_out, beta2_pow, beta2_pow_ptr, beta2_, dev_ctx);
      } else {
        const float* beta2_pow_data = beta2_pow.template data<float>();
        beta2_pow_out_p1 = dev_ctx.template Alloc<float>(beta2_pow_out);
        r = xpu::scale(dev_ctx.x_context(),
                       beta2_pow_data,
                       beta2_pow_out_p1,
                       beta2_pow.numel(),
                       false,
                       beta2_,
                       0.0f);
        xpu_wait(dev_ctx.x_context()->xpu_stream);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");
      }
    }
  }
  funcs::FreeData<float>(param, param_ptr);
  funcs::FreeData<float>(moment1, mom1_ptr);
  funcs::FreeData<float>(moment2, mom2_ptr);
  funcs::FreeData<float>(learning_rate, lr_ptr);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    adam, XPU, ALL_LAYOUT, phi::AdamDenseKernel, float, phi::dtype::float16) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
}
