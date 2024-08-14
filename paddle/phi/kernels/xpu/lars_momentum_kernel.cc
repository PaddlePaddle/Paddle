// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/lars_momentum_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void LarsMomentumKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& param,
    const std::vector<const DenseTensor*>& velocity,
    const std::vector<const DenseTensor*>& learning_rate,
    const std::vector<const DenseTensor*>& grad,
    const paddle::optional<std::vector<const DenseTensor*>>& master_param,
    const std::vector<float>& weight_decay_arr,
    float mu,
    float lars_coeff,
    float epsilon,
    bool multi_precision,
    float rescale_grad,
    std::vector<DenseTensor*> param_out,
    std::vector<DenseTensor*> velocity_out,
    std::vector<DenseTensor*> master_param_out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  std::vector<XPUType*> param_list;
  std::vector<XPUType*> grad_list;
  std::vector<XPUType*> param_out_list;
  std::vector<float*> velocity_list;
  std::vector<float*> velocity_out_list;
  std::vector<float*> lrs;
  std::vector<int> param_sizes;

  std::vector<float*> master_param_list;
  std::vector<float*> master_param_out_list;
  int op_num = param.size();
  for (int i = 0; i < op_num; ++i) {
    param_list.push_back(
        reinterpret_cast<XPUType*>(const_cast<T*>((param[i]->data<T>()))));
    grad_list.push_back(
        reinterpret_cast<XPUType*>(const_cast<T*>(grad[i]->data<T>())));
    param_out_list.push_back(
        reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(param_out[i])));
    velocity_list.push_back(const_cast<float*>(velocity[i]->data<float>()));
    velocity_out_list.push_back(dev_ctx.template Alloc<float>(velocity_out[i]));
    lrs.push_back(const_cast<float*>(learning_rate[i]->data<float>()));
    param_sizes.push_back(param[i]->numel());

    PADDLE_ENFORCE_EQ(
        param_list[i],
        param_out_list[i],
        common::errors::InvalidArgument(
            "Input(Param) and Output(ParamOut) must be the same Tensors."));
    PADDLE_ENFORCE_EQ(velocity_list[i],
                      velocity_out_list[i],
                      common::errors::InvalidArgument(
                          "Input(Velocity) and Output(VelocityOut) must be "
                          "the same Tensors."));
    if (multi_precision) {
      master_param_list.push_back(
          const_cast<float*>(master_param.get()[i]->data<float>()));
      master_param_out_list.push_back(
          dev_ctx.template Alloc<float>(master_param_out[i]));
      PADDLE_ENFORCE_EQ(master_param_list[i],
                        master_param_out_list[i],
                        common::errors::InvalidArgument(
                            "Input(MasterParam) and Output(MasterParamOut) "
                            "must be the same Tensors."));
    } else {
      master_param_list.push_back(nullptr);
      master_param_out_list.push_back(nullptr);
    }
  }

  int r = lars_momentum(dev_ctx.x_context(),
                        param_list,
                        grad_list,
                        velocity_list,
                        lrs,
                        master_param_list,
                        param_out_list,
                        velocity_out_list,
                        master_param_out_list,
                        weight_decay_arr,
                        param_sizes,
                        mu,
                        lars_coeff,
                        epsilon,
                        rescale_grad);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "lars_momentum");
}
}  // namespace phi

PD_REGISTER_KERNEL(lars_momentum,
                   XPU,
                   ALL_LAYOUT,
                   phi::LarsMomentumKernel,
                   float,
                   phi::dtype::float16) {}
