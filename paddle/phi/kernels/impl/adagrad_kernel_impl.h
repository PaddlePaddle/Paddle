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

#include "paddle/phi/kernels/adagrad_kernel.h"

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void AdagradDenseKernel(const Context& ctx,
                        const DenseTensor& param_t,
                        const DenseTensor& grad_t,
                        const DenseTensor& moment_t,
                        const DenseTensor& learning_rate,
                        float epsilon_t,
                        DenseTensor* param_out_tensor,
                        DenseTensor* moment_out_tensor) {
  ctx.template Alloc<T>(param_out_tensor);
  ctx.template Alloc<T>(moment_out_tensor);

  T epsilon = static_cast<T>(epsilon_t);

  auto param = EigenVector<T>::Flatten(param_t);

  auto grad = EigenVector<T>::Flatten(grad_t);

  auto moment = EigenVector<T>::Flatten(moment_t);

  auto param_out = EigenVector<T>::Flatten(*param_out_tensor);
  auto moment_out = EigenVector<T>::Flatten(*moment_out_tensor);
  auto place = *ctx.eigen_device();

  moment_out.device(place) = moment + grad * grad;
  Eigen::DSizes<int, 1> m_dsize(moment_out_tensor->numel());
  if (paddle::platform::is_cpu_place(ctx.GetPlace())) {
    auto* lr = learning_rate.data<T>();
    param_out.device(place) =
        param - lr[0] * grad / (moment_out.sqrt() + epsilon);
  } else {
    auto lr = EigenVector<T>::Flatten(learning_rate);
    param_out.device(place) =
        param - lr.broadcast(m_dsize) * grad / (moment_out.sqrt() + epsilon);
  }
}

}  // namespace phi
