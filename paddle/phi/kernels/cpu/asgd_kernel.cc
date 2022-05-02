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

#include "paddle/phi/kernels/asgd_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename Context>
void AsgdKernel(const Context& dev_ctx,
                const DenseTensor& param,
                const DenseTensor& grad,
                const DenseTensor& learning_rate,
                const DenseTensor& avg_param,
                const DenseTensor& current_step,
                float t0,
                DenseTensor* param_out,
                DenseTensor* avg_param_out,
                DenseTensor* current_step_out) {
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(avg_param_out);
  dev_ctx.template Alloc<T>(current_step_out);

  auto eigen_param = EigenVector<T>::Flatten(param);
  auto eigen_grad = EigenVector<T>::Flatten(grad);
  auto eigen_avg_param = EigenVector<T>::Flatten(avg_param);
  auto eigen_param_out = EigenVector<T>::Flatten(*param_out);
  auto eigen_avg_param_out = EigenVector<T>::Flatten(*avg_param_out);
  auto& place = *dev_ctx.eigen_device();

  auto lr = learning_rate.data<T>()[0];
  eigen_param_out.device(place) = eigen_param - lr * eigen_grad;

  T current_step_data = current_step.data<T>()[0];

  if (current_step_data <= t0) {
    eigen_avg_param_out.device(place) = eigen_param_out;
  } else {
    const auto mu1 = 1 / (current_step_data - t0);
    const auto mu2 = 1 - mu1;
    eigen_avg_param_out.device(place) =
        mu2 * eigen_avg_param + mu1 * eigen_param_out;
  }
  *current_step_out->mutable_data<T>(dev_ctx.GetPlace()) =
      current_step_data + 1;
}

}  // namespace phi

PD_REGISTER_KERNEL(asgd, CPU, ALL_LAYOUT, phi::AsgdKernel, float, double) {}
