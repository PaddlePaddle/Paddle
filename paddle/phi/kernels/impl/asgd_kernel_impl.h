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

#include "paddle/fluid/platform/place.h"
#include "paddle/phi/kernels/adadelta_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename Context>
void AsgdKernel(const Context& dev_ctx,
                const DenseTensor& param,
                const DenseTensor& learning_rate,
                const DenseTensor& grad,
                const DenseTensor& avg_param,
                const DenseTensor& current_step,
                const DenseTensor& t0,
                DenseTensor* param_out,
                DenseTensor* avg_param_out,
                DenseTensor* current_step_out) {
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(avg_param_out);

  auto eigen_param = EigenVector<T>::Flatten(param);
  auto eigen_grad = EigenVector<T>::Flatten(grad);
  auto eigen_avg_param = EigenVector<T>::Flatten(avg_param);
  auto eigen_current_step = EigenVector<T>::Flatten(current_step);
  auto eigen_t0 = EigenVector<T>::Flatten(t0);
  auto eigen_param_out = EigenVector<T>::Flatten(*param_out);
  auto eigen_avg_param_out = EigenVector<T>::Flatten(*avg_param_out);
  auto eigen_current_step_out = EigenVector<T>::Flatten(*current_step_out);
  auto& place = *dev_ctx.eigen_device();

  if (paddle::platform::is_cpu_place(dev_ctx.GetPlace())) {
    auto lr = learning_rate.data<T>()[0];
    eigen_param_out.device(place) = eigen_param - lr * eigen_grad;
  } else {
    Eigen::DSizes<int, 1> dsize(param_out->numel());
    auto eigen_lr = EigenVector<T>::Flatten(learning_rate);
    eigen_param_out.device(place) =
        eigen_param - eigen_lr.broadcast(dsize) * eigen_grad;
  }

  if (eigen_current_step < eigen_t0) {
    eigen_avg_param_out.device(place) = eigen_param_out;
  } else {
    // const auto mu = eigen_current_step - eigen_t0 + 1;
    eigen_avg_param_out.device(place) =
        eigen_avg_param + (eigen_param_out - eigen_avg_param) / eigen_current_step;
  }

  // eigen_current_step_out = eigen_current_step + 1;
  eigen_current_step++;
}

}  // namespace phi
