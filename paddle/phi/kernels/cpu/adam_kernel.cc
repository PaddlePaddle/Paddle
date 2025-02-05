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

#include <vector>

#include "glog/logging.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/funcs/jit/kernels.h"

PD_DECLARE_int32(inner_op_parallelism);

namespace phi {

template <typename T, typename Context>
void AdamDenseKernel(const Context& dev_ctx,
                     const DenseTensor& param,
                     const DenseTensor& grad,
                     const DenseTensor& learning_rate,
                     const DenseTensor& moment1,
                     const DenseTensor& moment2,
                     const paddle::optional<DenseTensor>& moment2_max,
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
                     bool amsgrad,
                     DenseTensor* param_out,
                     DenseTensor* moment1_out,
                     DenseTensor* moment2_out,
                     DenseTensor* moment2_max_out,
                     DenseTensor* beta1_pow_out,
                     DenseTensor* beta2_pow_out,
                     DenseTensor* master_param_outs) {
  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

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
  // skip_update=true, just copy input to output, and TensorCopy will call
  // mutable_data
  if (skip_update_) {
    VLOG(4) << "Adam skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    if (amsgrad) {
      phi::Copy(dev_ctx,
                moment2_max.get(),
                dev_ctx.GetPlace(),
                false,
                moment2_max_out);
    }
    if (!use_global_beta_pow) {
      phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
      phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    }
    return;
  }

  T beta1_ = beta1.to<T>();
  T beta2_ = beta2.to<T>();
  T epsilon_ = epsilon.to<T>();

  VLOG(3) << "beta1_pow.numel() : " << beta1_pow.numel();
  VLOG(3) << "beta2_pow.numel() : " << beta2_pow.numel();
  VLOG(3) << "param.numel(): " << param.numel();

  PADDLE_ENFORCE_EQ(
      beta1_pow_out->numel(),
      1,
      errors::InvalidArgument("beta1 pow output size should be 1, but received "
                              "value is:%d.",
                              beta1_pow_out->numel()));

  PADDLE_ENFORCE_EQ(
      beta2_pow_out->numel(),
      1,
      errors::InvalidArgument("beta2 pow output size should be 1, but received "
                              "value is:%d.",
                              beta2_pow_out->numel()));

  T beta1_p = beta1_pow.data<T>()[0];
  T beta2_p = beta2_pow.data<T>()[0];

  if (!use_global_beta_pow) {
    dev_ctx.template Alloc<T>(beta1_pow_out)[0] = beta1_ * beta1_p;
    dev_ctx.template Alloc<T>(beta2_pow_out)[0] = beta2_ * beta2_p;
  }

  T* param_out_ptr = dev_ctx.template Alloc<T>(param_out);
  T* mom1_out_ptr = dev_ctx.template Alloc<T>(moment1_out);
  T* mom2_out_ptr = dev_ctx.template Alloc<T>(moment2_out);
  T* mom2_max_out_ptr =
      amsgrad ? dev_ctx.template Alloc<T>(moment2_max_out) : nullptr;

  T learning_rate_ =
      learning_rate.data<T>()[0] * (sqrt(1 - beta2_p) / (1 - beta1_p));
  T eps = epsilon_ * sqrt(1 - beta2_p);

  phi::jit::adam_attr_t attr(beta1_, beta2_, amsgrad);
  int64_t numel = param.numel();

  const T* param_ptr = param.data<T>();
  const T* mom1_ptr = moment1.data<T>();
  const T* mom2_ptr = moment2.data<T>();
  const T* mom2_max_ptr = amsgrad ? moment2_max.get().data<T>() : nullptr;
  const T* grad_ptr = grad.data<T>();

  auto adam =
      phi::jit::KernelFuncs<phi::jit::AdamTuple<T>, phi::CPUPlace>::Cache().At(
          attr);

  static constexpr int64_t chunk_size = 512;

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < numel / chunk_size; ++i) {
    const int64_t offset = i * chunk_size;
    const T* mom2_max_in_data = amsgrad ? mom2_max_ptr + offset : nullptr;
    T* mom2_max_out_data = amsgrad ? mom2_max_out_ptr + offset : nullptr;

    adam(beta1_,
         beta2_,
         -learning_rate_,
         eps,
         chunk_size,
         grad_ptr + offset,
         mom1_ptr + offset,
         mom2_ptr + offset,
         mom2_max_in_data,
         param_ptr + offset,
         mom1_out_ptr + offset,
         mom2_out_ptr + offset,
         mom2_max_out_data,
         param_out_ptr + offset,
         amsgrad);
  }

  if (numel % chunk_size != 0) {
    const int64_t offset = (numel / chunk_size) * chunk_size;
    const int64_t tail_numel = numel % chunk_size;
    const T* mom2_max_in_data = amsgrad ? mom2_max_ptr + offset : nullptr;
    T* mom2_max_out_data = amsgrad ? mom2_max_out_ptr + offset : nullptr;

    adam(beta1_,
         beta2_,
         -learning_rate_,
         eps,
         tail_numel,
         grad_ptr + offset,
         mom1_ptr + offset,
         mom2_ptr + offset,
         mom2_max_in_data,
         param_ptr + offset,
         mom1_out_ptr + offset,
         mom2_out_ptr + offset,
         mom2_max_out_data,
         param_out_ptr + offset,
         amsgrad);
  }
}

template <typename T, typename Context>
void MergedAdamKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& param,
    const std::vector<const DenseTensor*>& grad,
    const std::vector<const DenseTensor*>& learning_rate,
    const std::vector<const DenseTensor*>& moment1,
    const std::vector<const DenseTensor*>& moment2,
    const paddle::optional<std::vector<const DenseTensor*>>& moment2_max,
    const std::vector<const DenseTensor*>& beta1_pow,
    const std::vector<const DenseTensor*>& beta2_pow,
    const paddle::optional<std::vector<const DenseTensor*>>& master_param,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad,
    std::vector<DenseTensor*> param_out,
    std::vector<DenseTensor*> moment1_out,
    std::vector<DenseTensor*> moment2_out,
    std::vector<DenseTensor*> moment2_max_out,
    std::vector<DenseTensor*> beta1_pow_out,
    std::vector<DenseTensor*> beta2_pow_out,
    std::vector<DenseTensor*> master_param_out) {
  size_t param_num = param.size();
  PADDLE_ENFORCE_EQ(
      param_num,
      grad.size(),
      errors::InvalidArgument("The size of Input(grad) must be equal to "
                              "Input(param), but got the size of Input(grad) "
                              "is %d, the size of Input(param) is %d.",
                              grad.size(),
                              param_num));
  PADDLE_ENFORCE_EQ(
      param_num,
      learning_rate.size(),
      errors::InvalidArgument(
          "The size of Input(learning_rate) must be equal to "
          "Input(param), but got the size of Input(learning_rate) "
          "is %d, the size of Input(param) is %d.",
          learning_rate.size(),
          param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    moment1.size(),
                    errors::InvalidArgument(
                        "The size of Input(moment1) must be equal to "
                        "Input(param), but got the size of Input(moment1) "
                        "is %d, the size of Input(param) is %d.",
                        moment1.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    moment2.size(),
                    errors::InvalidArgument(
                        "The size of Input(moment2) must be equal to "
                        "Input(param), but got the size of Input(moment2) "
                        "is %d, the size of Input(param) is %d.",
                        moment2.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta1_pow.size(),
                    errors::InvalidArgument(
                        "The size of Input(beta1_pow) must be equal to "
                        "Input(param), but got the size of Input(beta1_pow) "
                        "is %d, the size of Input(param) is %d.",
                        beta1_pow.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta2_pow.size(),
                    errors::InvalidArgument(
                        "The size of Input(beta2_pow) must be equal to "
                        "Input(param), but got the size of Input(beta2_pow) "
                        "is %d, the size of Input(param) is %d.",
                        beta2_pow.size(),
                        param_num));
  T beta1_ = beta1.to<T>();
  T beta2_ = beta2.to<T>();
  T epsilon_ = epsilon.to<T>();

  for (size_t idx = 0; idx < param_num; idx++) {
    const T* mom2_max_in_data =
        amsgrad ? moment2_max.get()[idx]->data<T>() : nullptr;
    T* mom2_max_out_data =
        amsgrad ? dev_ctx.template Alloc<T>(moment2_max_out[idx]) : nullptr;

    phi::funcs::AdamFunctor<T, phi::funcs::CPUAdam> functor(
        beta1_,
        beta2_,
        epsilon_,
        beta1_pow[idx]->data<T>(),
        beta2_pow[idx]->data<T>(),
        moment1[idx]->data<T>(),
        dev_ctx.template Alloc<T>(moment1_out[idx]),
        moment2[idx]->data<T>(),
        dev_ctx.template Alloc<T>(moment2_out[idx]),
        mom2_max_in_data,
        mom2_max_out_data,
        learning_rate[idx]->data<T>(),
        grad[idx]->data<T>(),
        param[idx]->data<T>(),
        dev_ctx.template Alloc<T>(param_out[idx]),
        amsgrad);
    functor(param[idx]->numel());
    if (!use_global_beta_pow) {
      dev_ctx.template Alloc<T>(beta1_pow_out[idx])[0] =
          beta1_ * beta1_pow[idx]->data<T>()[0];
      dev_ctx.template Alloc<T>(beta2_pow_out[idx])[0] =
          beta2_ * beta2_pow[idx]->data<T>()[0];
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(adam, CPU, ALL_LAYOUT, phi::AdamDenseKernel, float, double) {
}

PD_REGISTER_KERNEL(
    merged_adam, CPU, ALL_LAYOUT, phi::MergedAdamKernel, float, double) {}
