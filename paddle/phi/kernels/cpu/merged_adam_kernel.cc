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

#include "paddle/phi/kernels/merged_adam_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"

namespace phi {

template <typename T, typename Context>
void MergedAdamKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& param,
    const std::vector<const DenseTensor*>& grad,
    const std::vector<const DenseTensor*>& learning_rate,
    const std::vector<const DenseTensor*>& moment1,
    const std::vector<const DenseTensor*>& moment2,
    const std::vector<const DenseTensor*>& beta1_pow,
    const std::vector<const DenseTensor*>& beta2_pow,
    const paddle::optional<std::vector<const DenseTensor*>>& master_param,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<DenseTensor*> param_out,
    std::vector<DenseTensor*> moment1_out,
    std::vector<DenseTensor*> moment2_out,
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
        learning_rate[idx]->data<T>(),
        grad[idx]->data<T>(),
        param[idx]->data<T>(),
        dev_ctx.template Alloc<T>(param_out[idx]));
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

PD_REGISTER_KERNEL(
    merged_adam, CPU, ALL_LAYOUT, phi::MergedAdamKernel, float, double) {}
