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

#include "paddle/phi/kernels/multi_tensor_adam_kernel.h"
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "paddle/phi/kernels/adam_kernel.h"
#include "paddle/phi/kernels/adamw_kernel.h"

namespace phi {

static paddle::optional<DenseTensor> TensorPtrToOptionalTensor(
    const paddle::optional<std::vector<const DenseTensor*>>& t, size_t idx) {
  return t ? paddle::optional<DenseTensor>(*(t.get()[idx])) : paddle::none;
}

template <typename T, typename Context>
void MultiTensorAdamKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& params,
    const std::vector<const DenseTensor*>& grads,
    const DenseTensor& learning_rate,
    const std::vector<const DenseTensor*>& moments1,
    const std::vector<const DenseTensor*>& moments2,
    const std::vector<const DenseTensor*>& beta1_pow,
    const std::vector<const DenseTensor*>& beta2_pow,
    const paddle::optional<std::vector<const DenseTensor*>>& master_params,
    const paddle::optional<DenseTensor>& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    int chunk_size,
    float weight_decay,
    bool use_adamw,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<DenseTensor*> params_out,
    std::vector<DenseTensor*> moments1_out,
    std::vector<DenseTensor*> moments2_out,
    std::vector<DenseTensor*> beta1_pow_out,
    std::vector<DenseTensor*> beta2_pow_out,
    std::vector<DenseTensor*> master_params_out) {
  size_t params_num = params.size();
  PADDLE_ENFORCE_EQ(
      params_num,
      grads.size(),
      errors::InvalidArgument("The size of Input(grads) must be equal to "
                              "Input(params), but got the size of Input(grads) "
                              "is %d, the size of Input(params) is %d.",
                              grads.size(),
                              params_num));
  PADDLE_ENFORCE_EQ(params_num,
                    moments1.size(),
                    errors::InvalidArgument(
                        "The size of Input(moments1) must be equal to "
                        "Input(param), but got the size of Input(moments1) "
                        "is %d, the size of Input(param) is %d.",
                        moments1.size(),
                        params_num));
  PADDLE_ENFORCE_EQ(params_num,
                    moments2.size(),
                    errors::InvalidArgument(
                        "The size of Input(moments2) must be equal to "
                        "Input(param), but got the size of Input(moments2) "
                        "is %d, the size of Input(param) is %d.",
                        moments2.size(),
                        params_num));
  PADDLE_ENFORCE_EQ(params_num,
                    beta1_pow.size(),
                    errors::InvalidArgument(
                        "The size of Input(beta1_pow) must be equal to "
                        "Input(param), but got the size of Input(beta1_pow) "
                        "is %d, the size of Input(param) is %d.",
                        beta1_pow.size(),
                        params_num));
  PADDLE_ENFORCE_EQ(params_num,
                    beta2_pow.size(),
                    errors::InvalidArgument(
                        "The size of Input(beta2_pow) must be equal to "
                        "Input(param), but got the size of Input(beta2_pow) "
                        "is %d, the size of Input(param) is %d.",
                        beta2_pow.size(),
                        params_num));

  for (size_t idx = 0; idx < params_num; idx++) {
    paddle::optional<DenseTensor> master_params_tmp =
        TensorPtrToOptionalTensor(master_params, idx);
    if (!use_adamw) {
      AdamDenseKernel<T, Context>(
          dev_ctx,
          *params[idx],
          *grads[idx],
          learning_rate,
          *moments1[idx],
          *moments2[idx],
          *beta1_pow[idx],
          *beta2_pow[idx],
          master_params_tmp,
          skip_update,
          beta1,
          beta2,
          epsilon,
          false,
          1000,
          multi_precision,
          use_global_beta_pow,
          params_out[idx],
          moments1_out[idx],
          moments2_out[idx],
          beta1_pow_out[idx],
          beta2_pow_out[idx],
          master_params_out.empty() ? nullptr : master_params_out[idx]);
    } else {
      AdamwDenseKernel<T, Context>(
          dev_ctx,
          *params[idx],
          *grads[idx],
          learning_rate,
          *moments1[idx],
          *moments2[idx],
          *beta1_pow[idx],
          *beta2_pow[idx],
          master_params_tmp,
          skip_update,
          beta1,
          beta2,
          epsilon,
          1.0,
          weight_decay,
          use_adamw,
          false,
          1000,
          multi_precision,
          use_global_beta_pow,
          params_out[idx],
          moments1_out[idx],
          moments2_out[idx],
          beta1_pow_out[idx],
          beta2_pow_out[idx],
          master_params_out.empty() ? nullptr : master_params_out[idx]);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(multi_tensor_adam,
                   CPU,
                   ALL_LAYOUT,
                   phi::MultiTensorAdamKernel,
                   float,
                   double) {}
