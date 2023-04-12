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

#pragma once
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/adadelta_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void AdadeltaKernel(const Context& dev_ctx,
                    const DenseTensor& param,
                    const DenseTensor& grad,
                    const DenseTensor& avg_squared_grad,
                    const DenseTensor& avg_squared_update,
                    const DenseTensor& learning_rate,
                    const paddle::optional<DenseTensor>& master_param,
                    float rho,
                    float epsilon,
                    bool multi_precision,
                    DenseTensor* param_out,
                    DenseTensor* avg_squared_grad_out,
                    DenseTensor* avg_squared_update_out,
                    DenseTensor* master_param_outs) {
  using MPDType = typename phi::dtype::template MPTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<MPDType>(avg_squared_grad_out);
  dev_ctx.template Alloc<MPDType>(avg_squared_update_out);

  MPDType rho_ = static_cast<MPDType>(rho);
  MPDType epsilon_ = static_cast<MPDType>(epsilon);

  auto eigen_param = EigenVector<T>::Flatten(param);
  auto eigen_grad = EigenVector<T>::Flatten(grad);
  // Squared gradient accumulator
  auto eigen_avg_squared_grad = EigenVector<MPDType>::Flatten(avg_squared_grad);
  // Squared updates accumulator
  auto eigen_avg_squared_update =
      EigenVector<MPDType>::Flatten(avg_squared_update);
  auto eigen_param_out = EigenVector<T>::Flatten(*param_out);
  auto eigen_avg_squared_grad_out =
      EigenVector<MPDType>::Flatten(*avg_squared_grad_out);
  auto eigen_avg_squared_update_out =
      EigenVector<MPDType>::Flatten(*avg_squared_update_out);
  auto& place = *dev_ctx.eigen_device();
  auto eigen_grad_cast = eigen_grad.template cast<MPDType>();
  eigen_avg_squared_grad_out.device(place) =
      rho_ * eigen_avg_squared_grad + (1 - rho_) * eigen_grad_cast.square();
  auto update =
      -(((eigen_avg_squared_update + epsilon_).sqrt()) /
        ((eigen_avg_squared_grad_out + epsilon_).sqrt()) * eigen_grad_cast);
  Eigen::DSizes<int, 1> m_dsize(avg_squared_update_out->numel());
  auto lr = EigenVector<MPDType>::Flatten(learning_rate);
  if (multi_precision) {
    auto eigen_master_param_out =
        EigenVector<MPDType>::Flatten(*master_param_outs);
    auto eigen_master_param = EigenVector<MPDType>::Flatten(*master_param);

    eigen_master_param_out.device(place) =
        eigen_master_param + lr.broadcast(m_dsize) * update;
    eigen_param_out.device(place) =
        (eigen_param.template cast<MPDType>() + lr.broadcast(m_dsize) * update)
            .template cast<T>();
  } else {
    eigen_param_out.device(place) =
        eigen_param + (lr.broadcast(m_dsize) * update).template cast<T>();
  }
  eigen_avg_squared_update_out.device(place) =
      rho_ * eigen_avg_squared_update + (1 - rho_) * update.square();
}

}  // namespace phi
