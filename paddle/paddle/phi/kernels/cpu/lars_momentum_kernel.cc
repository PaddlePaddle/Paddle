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
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

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
  int op_num = static_cast<int>(param.size());
  T mu_ = static_cast<T>(mu);
  for (int i = 0; i < op_num; ++i) {
    auto* lr = learning_rate[i]->data<T>();
    T lars_weight_decay = weight_decay_arr[i];
    dev_ctx.template Alloc<T>(param_out[i]);
    dev_ctx.template Alloc<T>(velocity_out[i]);

    auto p_out = phi::EigenVector<T>::Flatten(*(param_out[i]));
    auto v_out = phi::EigenVector<T>::Flatten(*(velocity_out[i]));
    auto p = phi::EigenVector<T>::Flatten(*(param[i]));
    auto v = phi::EigenVector<T>::Flatten(*(velocity[i]));
    Eigen::TensorMap<Eigen::Tensor<const T, 1, 1>> g =
        phi::EigenVector<T>::Flatten(*(grad[i]));
    auto rescale_g = static_cast<T>(rescale_grad) * g;

    phi::DenseTensor p_norm_t, g_norm_t;
    p_norm_t.Resize({1});
    g_norm_t.Resize({1});
    dev_ctx.template Alloc<T>(&p_norm_t);
    dev_ctx.template Alloc<T>(&g_norm_t);
    auto ep_norm = phi::EigenScalar<T>::From(p_norm_t);
    auto eg_norm = phi::EigenScalar<T>::From(g_norm_t);
    ep_norm = p.square().sum().sqrt();
    eg_norm = rescale_g.square().sum().sqrt();

    T local_lr = lr[0];
    if (lars_weight_decay > 0 && ep_norm(0) > 0 && eg_norm(0) > 0) {
      local_lr = lr[0] * lars_coeff * ep_norm(0) /
                 (eg_norm(0) + lars_weight_decay * ep_norm(0) + epsilon);
    }
    v_out = v * mu_ + local_lr * (rescale_g + lars_weight_decay * p);
    p_out = p - v_out;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    lars_momentum, CPU, ALL_LAYOUT, phi::LarsMomentumKernel, float, double) {}
