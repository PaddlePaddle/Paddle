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

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename Context>
float GetAbsMax(const Context& dev_ctx,
                const float* input,
                float* buffer_xpu,
                int64_t numel) {
  float buffer_cpu[6];
  // int findmax(Context* ctx, const T* x, float* maxptr, int64_t len);
  int r = xpu::findmax<float>(dev_ctx.x_context(), input, buffer_xpu, numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax");
  memory_utils::Copy(CPUPlace(),
                     static_cast<void*>(buffer_cpu),
                     dev_ctx.GetPlace(),
                     static_cast<void*>(buffer_xpu),
                     sizeof(float) * 6);
  float* max_value = std::max_element(buffer_cpu, buffer_cpu + 6);
  return *max_value;
}

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
  // check moment_dtype
  auto moment1_dtype = moment1.dtype();
  auto moment2_dtype = moment2.dtype();
  PADDLE_ENFORCE_EQ(moment1_dtype,
                    moment1_out->dtype(),
                    errors::InvalidArgument(
                        "moment1.dtype does not match moment1_out->dtype"));
  PADDLE_ENFORCE_EQ(moment2_dtype,
                    moment2_out->dtype(),
                    errors::InvalidArgument(
                        "moment2.dtype does not match moment2_out->dtype"));
  PADDLE_ENFORCE_EQ(
      moment1_dtype,
      moment2_dtype,
      errors::InvalidArgument("moment1.dtype does not match moment2.dtype"));

  bool moment_in_fp16 = false;
  if (moment1_dtype == phi::DataType::FLOAT16) {
    moment_in_fp16 = true;
  } else {
    PADDLE_ENFORCE_EQ(
        moment1_dtype,
        phi::DataType::FLOAT32,
        errors::InvalidArgument("moment1.dtype is neither fp32 nor fp16"));
  }

  float* moment1_input_for_xdnn = nullptr;
  float* moment2_input_for_xdnn = nullptr;
  float* moment1_output_for_xdnn = nullptr;
  float* moment2_output_for_xdnn = nullptr;

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  if (moment_in_fp16) {
    // allocate temp buffer on XPU
    moment1_input_for_xdnn = RAII_GUARD.alloc_l3_or_gm<float>(moment1.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment1_input_for_xdnn);
    moment2_input_for_xdnn = RAII_GUARD.alloc_l3_or_gm<float>(moment2.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment2_input_for_xdnn);
    moment1_output_for_xdnn =
        RAII_GUARD.alloc_l3_or_gm<float>(moment1_out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment1_output_for_xdnn);
    moment2_output_for_xdnn =
        RAII_GUARD.alloc_l3_or_gm<float>(moment2_out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(moment2_output_for_xdnn);

    int r = 0;
    using XPUType16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

    // cast moment1 and moment2, from fp16 to fp32
    // int cast(Context* ctx, const TX* x, TY* y, int64_t len);
    r = xpu::cast<XPUType16, float>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType16*>(
            moment1.template data<phi::dtype::float16>()),
        moment1_input_for_xdnn,
        moment1.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment1 from fp16 to float");
    r = xpu::cast<XPUType16, float>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType16*>(
            moment2.template data<phi::dtype::float16>()),
        moment2_input_for_xdnn,
        moment2.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment2 from fp16 to float");

    // de-scale using meta's scale_value
    // int scale(Context* ctx, const T* x, T* y, int64_t len, bool
    // bias_after_scale, float _scale, float _bias);
    phi::DenseTensorMeta moment1_meta = moment1.meta();
    if (moment1_meta.scale_value > 0) {
      r = xpu::scale<float>(dev_ctx.x_context(),
                            moment1_input_for_xdnn,
                            moment1_input_for_xdnn,
                            moment1.numel(),
                            false,
                            1.0f / moment1_meta.scale_value,
                            0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "de-scale for moment1");
    }
    phi::DenseTensorMeta moment2_meta = moment2.meta();
    if (moment2_meta.scale_value > 0) {
      r = xpu::scale<float>(dev_ctx.x_context(),
                            moment2_input_for_xdnn,
                            moment2_input_for_xdnn,
                            moment2.numel(),
                            false,
                            1.0f / moment2_meta.scale_value,
                            0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "de-scale for moment2");
    }
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    std::vector<bool> skip_update_vec;
    phi::TensorToVector(*skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }
  if (skip_update_) {
    VLOG(4) << "Adamw skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    if (!use_global_beta_pow) {
      phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
      phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    }
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
  if (!with_decay) {
    coeff = static_cast<float>(0.0);
  }

  float* new_lr = RAII_GUARD.alloc_l3_or_gm<float>(learning_rate.numel());
  PADDLE_ENFORCE_XDNN_NOT_NULL(new_lr);
  int r = 0;
  r = xpu::scale(dev_ctx.x_context(),
                 learning_rate.template data<float>(),
                 new_lr,
                 learning_rate.numel(),
                 false,
                 lr_ratio,
                 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");

  // int adamw(Context* ctx, const T* g, const float* mom1, const float* mom2,
  // const T* param, const float* beta1_pow, const float* beta2_pow, const
  // float* lr, float* moment1_out, float* moment2_out, T* param_out, float
  // beta1, float beta2, float epsilon, float coeff, int64_t n);
  r = xpu::adamw(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(grad.template data<T>()),
      moment_in_fp16 ? moment1_input_for_xdnn : moment1.template data<float>(),
      moment_in_fp16 ? moment2_input_for_xdnn : moment2.template data<float>(),
      reinterpret_cast<const XPUType*>(param.template data<T>()),
      beta1_pow_ptr,
      beta2_pow_ptr,
      new_lr,
      moment_in_fp16 ? moment1_output_for_xdnn
                     : dev_ctx.template Alloc<float>(moment1_out),
      moment_in_fp16 ? moment2_output_for_xdnn
                     : dev_ctx.template Alloc<float>(moment2_out),
      reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(param_out)),
      beta1_,
      beta2_,
      epsilon_,
      coeff,
      param.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "adamw");

  if (moment_in_fp16) {
    int r = 0;
    using XPUType16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

    // findmax and calculate scale_value for moment1 and moment2
    float* buffer_for_findmax = RAII_GUARD.alloc_l3_or_gm<float>(6);

    // for moment1
    float moment1_max = GetAbsMax<Context>(dev_ctx,
                                           moment1_output_for_xdnn,
                                           buffer_for_findmax,
                                           moment1_out->numel());
    float moment1_scale_value = 65504.0f / moment1_max / 2.0f;
    // int scale(Context* ctx, const T* x, T* y, int64_t len, bool
    // bias_after_scale, float _scale, float _bias);
    r = xpu::scale<float>(dev_ctx.x_context(),
                          moment1_output_for_xdnn,
                          moment1_output_for_xdnn,
                          moment1_out->numel(),
                          false,
                          moment1_scale_value,
                          0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(
        r, "scale before convert to fp16, for moment1_output_for_xdnn");
    // write to meta info
    phi::DenseTensorMeta moment1_out_meta = moment1_out->meta();
    moment1_out_meta.scale_value = moment1_scale_value;
    moment1_out->set_meta(moment1_out_meta);

    // for moment2
    float moment2_max = GetAbsMax<Context>(dev_ctx,
                                           moment2_output_for_xdnn,
                                           buffer_for_findmax,
                                           moment2_out->numel());
    float moment2_scale_value = 65504.0f / moment2_max / 2.0f;
    // int scale(Context* ctx, const T* x, T* y, int64_t len, bool
    // bias_after_scale, float _scale, float _bias);
    r = xpu::scale<float>(dev_ctx.x_context(),
                          moment2_output_for_xdnn,
                          moment2_output_for_xdnn,
                          moment2_out->numel(),
                          false,
                          moment2_scale_value,
                          0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(
        r, "scale before convert to fp16, for moment2_output_for_xdnn");
    // write to meta info
    phi::DenseTensorMeta moment2_out_meta = moment2_out->meta();
    moment2_out_meta.scale_value = moment2_scale_value;
    moment2_out->set_meta(moment2_out_meta);

    // cast moment1 and moment2 output, from fp32 to fp16
    // int cast(Context* ctx, const TX* x, TY* y, int64_t len);
    r = xpu::cast<float, XPUType16>(
        dev_ctx.x_context(),
        moment1_output_for_xdnn,
        reinterpret_cast<XPUType16*>(
            dev_ctx.template Alloc<phi::dtype::float16>(moment1_out)),
        moment1.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment1_out from float to fp16");
    r = xpu::cast<float, XPUType16>(
        dev_ctx.x_context(),
        moment2_output_for_xdnn,
        reinterpret_cast<XPUType16*>(
            dev_ctx.template Alloc<phi::dtype::float16>(moment2_out)),
        moment2.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast moment2_out from float to fp16");
  }

  if (!use_global_beta_pow) {
    // update in cpu
    if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
      const float* beta1_pow_p = beta1_pow.template data<float>();
      dev_ctx.template HostAlloc<float>(beta1_pow_out)[0] =
          beta1_ * beta1_pow_p[0];
      const float* beta2_pow_p = beta2_pow.template data<float>();
      dev_ctx.template HostAlloc<float>(beta2_pow_out)[0] =
          beta2_ * beta2_pow_p[0];
      xpu_wait(dev_ctx.x_context()->xpu_stream);
    } else {  // update in  xpu
      float* beta1_pow_out_p = dev_ctx.template Alloc<float>(beta1_pow_out);
      float* beta2_pow_out_p = dev_ctx.template Alloc<float>(beta2_pow_out);
      int r = xpu::scale(dev_ctx.x_context(),
                         beta1_pow_ptr,
                         beta1_pow_out_p,
                         beta1_pow.numel(),
                         false,
                         beta1_,
                         0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
      r = xpu::scale(dev_ctx.x_context(),
                     beta2_pow_ptr,
                     beta2_pow_out_p,
                     beta2_pow.numel(),
                     false,
                     beta2_,
                     0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    adamw, XPU, ALL_LAYOUT, phi::AdamwDenseKernel, float, phi::dtype::float16) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->OutputAt(3)
      .SetBackend(phi::Backend::UNDEFINED)
      .SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(4)
      .SetBackend(phi::Backend::UNDEFINED)
      .SetDataType(phi::DataType::FLOAT32);
}
