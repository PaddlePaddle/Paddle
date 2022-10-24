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

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "paddle/phi/kernels/adam_kernel.h"
#include "paddle/phi/kernels/adamw_kernel.h"

DECLARE_int32(inner_op_parallelism);

namespace phi {

paddle::optional<DenseTensor> TensorPtrToOptionalTensor(const DenseTensor* t) {
  if (t != nullptr) {
    return *t;
  } else {
    return paddle::none;
  }
}

template <typename T, typename Context>
void MultiTensorAdamKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& params,
    const std::vector<const DenseTensor*>& grads,
    const std::vector<const DenseTensor*>& moments1,
    const std::vector<const DenseTensor*>& moments2,
    const paddle::optional<std::vector<const DenseTensor*>>& master_param,
    const DenseTensor& beta1_pow,
    const DenseTensor& beta2_pow,
    const DenseTensor& learning_rate,
    const paddle::optional<DenseTensor>& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    int compute_group_size,
    float weight_decay,
    bool mode,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<DenseTensor*> params_out,
    std::vector<DenseTensor*> moments1_out,
    std::vector<DenseTensor*> moments2_out,
    std::vector<DenseTensor*> master_param_out,
    DenseTensor* beta1_pow_out,
    DenseTensor* beta2_pow_out) {
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

  for (size_t idx = 0; idx < params_num; idx++) {
    paddle::optional<DenseTensor> master_param_tmp = paddle::none;
    if (master_param) {
      master_param_tmp = TensorPtrToOptionalTensor(master_param.get()[idx]);
    }
    if (!mode) {
      AdamDenseKernel<T, Context>(
          dev_ctx,
          *params[idx],
          *grads[idx],
          learning_rate,
          *moments1[idx],
          *moments2[idx],
          beta1_pow,
          beta2_pow,
          master_param_tmp,
          skip_update,
          beta1,
          beta2,
          epsilon,
          false,
          1000,
          multi_precision,
          true,
          params_out[idx],
          moments1_out[idx],
          moments2_out[idx],
          beta1_pow_out,
          beta2_pow_out,
          master_param_out.empty() ? nullptr : master_param_out[idx]);
    } else {
      AdamwDenseKernel<T, Context>(
          dev_ctx,
          *params[idx],
          *grads[idx],
          learning_rate,
          *moments1[idx],
          *moments2[idx],
          beta1_pow,
          beta2_pow,
          master_param_tmp,
          skip_update,
          beta1,
          beta2,
          epsilon,
          1.0,
          weight_decay,
          mode,
          false,
          1000,
          multi_precision,
          true,
          params_out[idx],
          moments1_out[idx],
          moments2_out[idx],
          beta1_pow_out,
          beta2_pow_out,
          master_param_out.empty() ? nullptr : master_param_out[idx]);
    }
  }

  T beta1_ = beta1.to<T>();
  T beta2_ = beta2.to<T>();
  if (!use_global_beta_pow) {
    dev_ctx.template Alloc<T>(beta1_pow_out)[0] =
        beta1_ * beta1_pow.data<T>()[0];
    dev_ctx.template Alloc<T>(beta2_pow_out)[0] =
        beta2_ * beta2_pow.data<T>()[0];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(multi_tensor_adam,
                   CPU,
                   ALL_LAYOUT,
                   phi::MultiTensorAdamKernel,
                   float,
                   double) {}
