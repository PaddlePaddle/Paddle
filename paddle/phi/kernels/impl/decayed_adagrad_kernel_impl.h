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

#pragma once

#include "paddle/phi/kernels/decayed_adagrad_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void DecayedAdagradDenseKernel(const Context& dev_ctx,
                               const DenseTensor& param_t,
                               const DenseTensor& grad_t,
                               const DenseTensor& moment_t,
                               const DenseTensor& learning_rate,
                               float decay,
                               float epsilon,
                               DenseTensor* param_out_t,
                               DenseTensor* moment_out_t) {
  dev_ctx.template Alloc<T>(param_out_t);
  dev_ctx.template Alloc<T>(moment_out_t);

  auto param = EigenVector<T>::Flatten(param_t);
  auto grad = EigenVector<T>::Flatten(grad_t);
  auto moment = EigenVector<T>::Flatten(moment_t);
  auto lr = EigenVector<T>::Flatten(learning_rate);

  auto param_out = EigenVector<T>::Flatten(*param_out_t);
  auto moment_out = EigenVector<T>::Flatten(*moment_out_t);
  auto& place = *dev_ctx.eigen_device();

  moment_out.device(place) = decay * moment + (1 - decay) * grad * grad;
  Eigen::DSizes<int, 1> m_dsize(moment_out_t->numel());
  param_out.device(place) =
      param - lr.broadcast(m_dsize) * grad / (moment_out.sqrt() + epsilon);
}
}  // namespace phi
