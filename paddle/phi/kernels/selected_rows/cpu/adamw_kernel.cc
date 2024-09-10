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

#include "paddle/phi/kernels/selected_rows/adamw_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/adam_kernel.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/selected_rows/adam_kernel.h"

namespace phi::sr {

template <typename T, typename Context>
void AdamwDenseParamSparseGradKernel(
    const Context& dev_ctx,
    const DenseTensor& param,
    const SelectedRows& grad,
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
    float lr_ratio,
    float coeff,
    bool with_decay,
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
  VLOG(3) << "Skip update" << skip_update_;

  if (skip_update_ || !with_decay) {
    AdamDenseParamSparseGradKernel<T, Context>(dev_ctx,
                                               param,
                                               grad,
                                               learning_rate,
                                               moment1,
                                               moment2,
                                               moment2_max,
                                               beta1_pow,
                                               beta2_pow,
                                               master_param,
                                               skip_update,
                                               beta1,
                                               beta2,
                                               epsilon,
                                               lazy_mode,
                                               min_row_size_to_use_multithread,
                                               multi_precision,
                                               use_global_beta_pow,
                                               amsgrad,
                                               param_out,
                                               moment1_out,
                                               moment2_out,
                                               moment2_max_out,
                                               beta1_pow_out,
                                               beta2_pow_out,
                                               master_param_outs);
    return;
  }

  auto* param_ =
      master_param.is_initialized() ? master_param.get_ptr() : &param;
  T coeff_ = static_cast<T>(coeff);
  T lr_ratio_ = static_cast<T>(lr_ratio);
  funcs::AdamWFunctor<T, funcs::CPUAdamW> functor(
      coeff_,
      lr_ratio_,
      learning_rate.data<T>(),
      const_cast<T*>(param_->data<T>()));
  functor(param_->numel());

  AdamDenseParamSparseGradKernel<T, Context>(dev_ctx,
                                             param,
                                             grad,
                                             learning_rate,
                                             moment1,
                                             moment2,
                                             moment2_max,
                                             beta1_pow,
                                             beta2_pow,
                                             master_param,
                                             skip_update,
                                             beta1,
                                             beta2,
                                             epsilon,
                                             lazy_mode,
                                             min_row_size_to_use_multithread,
                                             multi_precision,
                                             use_global_beta_pow,
                                             amsgrad,
                                             param_out,
                                             moment1_out,
                                             moment2_out,
                                             moment2_max_out,
                                             beta1_pow_out,
                                             beta2_pow_out,
                                             master_param_outs);
}

}  // namespace phi::sr

PD_REGISTER_KERNEL(adamw_dense_param_sparse_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::AdamwDenseParamSparseGradKernel,
                   float,
                   double) {}
