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

#include "paddle/phi/kernels/adadelta_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename Context>
void AdadeltaKernel(const Context& dev_ctx,
                    const DenseTensor& param,
                    const DenseTensor& grad,
                    const DenseTensor& avg_squared_grad,
                    const DenseTensor& avg_squared_update,
                    float rho,
                    float epsilon,
                    DenseTensor* param_out,
                    DenseTensor* avg_squared_grad_out,
                    DenseTensor* avg_squared_update_out) {
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(avg_squared_grad_out);
  dev_ctx.template Alloc<T>(avg_squared_update_out);

  T rho_ = static_cast<T>(rho);
  T epsilon_ = static_cast<T>(epsilon);

  auto eigen_param = EigenVector<T>::Flatten(param);
  auto eigen_grad = EigenVector<T>::Flatten(grad);
  // Squared gradient accumulator
  auto eigen_avg_squared_grad = EigenVector<T>::Flatten(avg_squared_grad);
  // Squared updates accumulator
  auto eigen_avg_squared_update = EigenVector<T>::Flatten(avg_squared_update);
  auto eigen_param_out = EigenVector<T>::Flatten(*param_out);
  auto eigen_avg_squared_grad_out =
      EigenVector<T>::Flatten(*avg_squared_grad_out);
  auto eigen_avg_squared_update_out =
      EigenVector<T>::Flatten(*avg_squared_update_out);
  auto& place = *dev_ctx.eigen_device();

  eigen_avg_squared_grad_out.device(place) =
      rho_ * eigen_avg_squared_grad + (1 - rho_) * eigen_grad.square();
  auto update = -((eigen_avg_squared_update + epsilon_) /
                  (eigen_avg_squared_grad_out + epsilon_))
                     .sqrt() *
                eigen_grad;
  eigen_avg_squared_update_out.device(place) =
      rho_ * eigen_avg_squared_update + (1 - rho_) * update.square();
  eigen_param_out.device(place) = eigen_param + update;
}

}  // namespace phi
