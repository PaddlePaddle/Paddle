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

#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

#include "paddle/phi/kernels/merged_momentum_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MergedMomentumKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& params,
    const std::vector<const DenseTensor*>& grad,
    const std::vector<const DenseTensor*>& velocity,
    const std::vector<const DenseTensor*>& learning_rate,
    const paddle::optional<std::vector<const DenseTensor*>>& master_param,
    float mu_in,
    bool use_nesterov,
    const std::vector<std::string>& regularization_method,
    const std::vector<float>& regularization_coeff,
    bool multi_precision,
    float rescale_grad,
    std::vector<DenseTensor*> params_out,
    std::vector<DenseTensor*> velocity_out,
    std::vector<DenseTensor*> master_param_out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto lr = learning_rate[0];
  T mu = static_cast<T>(mu_in);
  int op_num = params.size();
  PADDLE_ENFORCE_EQ(op_num,
                    params_out.size(),
                    errors::InvalidArgument(
                        "The size of Output(ParamOut) must be equal to "
                        "Input(Param), but got the size of Output(ParamOut) "
                        "is %d, the size of Input(Param) is %d.",
                        params_out.size(),
                        op_num));
  PADDLE_ENFORCE_EQ(op_num,
                    velocity.size(),
                    errors::InvalidArgument(
                        "The size of Output(Velocity) must be equal to "
                        "Input(Param), but got the size of Output(Velocity) "
                        "is %d, the size of Input(Param) is %d.",
                        velocity.size(),
                        op_num));
  PADDLE_ENFORCE_EQ(op_num,
                    velocity_out.size(),
                    errors::InvalidArgument(
                        "The size of Output(VelocityOut) must be equal to "
                        "Input(Param), but got the size of Output(VelocityOut) "
                        "is %d, the size of Input(Param) is %d.",
                        velocity_out.size(),
                        op_num));
  PADDLE_ENFORCE_EQ(
      op_num,
      grad.size(),
      errors::InvalidArgument(
          "The size of Input(Grad) must be equal to Input(Param), but got "
          "the size of Input(Grad) is %d, the size of Input(Param) is %d.",
          grad.size(),
          op_num));
  std::vector<XPUType*> param_list(op_num);
  std::vector<XPUType*> velocity_list(op_num);
  std::vector<XPUType*> grad_list(op_num);
  std::vector<XPUType*> velocity_out_list(op_num);
  std::vector<XPUType*> param_out_list(op_num);
  std::vector<int> sizes(op_num);
  std::vector<float> l2_weight_decay(op_num);
  if (op_num > 0) {
    for (int j = 0; j < op_num; j++) {
      param_list[j] =
          reinterpret_cast<XPUType*>(const_cast<T*>(params[j]->data<T>()));
      velocity_list[j] =
          reinterpret_cast<XPUType*>(const_cast<T*>(velocity[j]->data<T>()));
      grad_list[j] =
          reinterpret_cast<XPUType*>(const_cast<T*>(grad[j]->data<T>()));
      param_out_list[j] = reinterpret_cast<XPUType*>(params_out[j]->data<T>());
      velocity_out_list[j] =
          reinterpret_cast<XPUType*>(velocity_out[j]->data<T>());
      sizes[j] = static_cast<int>(params[j]->numel());
      if (regularization_method[j] != "l2_decay") {
        l2_weight_decay[j] = 0.0f;
      } else {
        l2_weight_decay[j] = static_cast<float>(regularization_coeff[j]);
      }
      PADDLE_ENFORCE_EQ(params[j],
                        params_out[j],
                        errors::InvalidArgument(
                            "The size of Input(Param) and Output(ParamOut) "
                            "must be the same Tensors."));
      PADDLE_ENFORCE_EQ(velocity[j],
                        velocity_out[j],
                        errors::InvalidArgument(
                            "The size of Input(velocity) and Output(velocity) "
                            "must be the same Tensors."));
    }
  } else {
    return;
  }
  int r = xpu::merged_momentum(dev_ctx.x_context(),
                               param_list,
                               velocity_list,
                               grad_list,
                               param_out_list,
                               velocity_out_list,
                               l2_weight_decay,
                               sizes,
                               lr->data<float>(),
                               mu,
                               use_nesterov);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "merged_momentum");
}

}  // namespace phi

PD_REGISTER_KERNEL(merged_momentum,
                   XPU,
                   ALL_LAYOUT,
                   phi::MergedMomentumKernel,
                   float,
                   phi::dtype::float16) {}
