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

#include "paddle/phi/kernels/rprop_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/jit/kernels.h"

namespace phi {

template <typename T>
void RpropKernelCPUImpl(const DenseTensor& param,
                        const DenseTensor& grad,
                        const DenseTensor& prev,
                        const DenseTensor& learning_rate,
                        float delta_min,
                        float delta_max,
                        float eta_negative,
                        float eta_positive,
                        DenseTensor* param_out,
                        DenseTensor* prev_out,
                        DenseTensor* learning_rate_out) {
  auto param_eigen = EigenVector<T>::Flatten(param);
  DenseTensor grad_new(grad);
  auto grad_eigen = EigenVector<T>::Flatten(grad_new);
  DenseTensor grad_new_new(grad);
  auto grad_sign_eigen = EigenVector<T>::Flatten(grad_new_new);
  auto prev_eigen = EigenVector<T>::Flatten(prev);
  DenseTensor learning_rate_new(learning_rate);
  auto learning_rate_eigen = EigenVector<T>::Flatten(learning_rate_new);
  DenseTensor learning_rate_new_new(learning_rate);
  auto eta_eigen = EigenVector<T>::Flatten(learning_rate_new_new);
  auto param_out_eigen = EigenVector<T>::Flatten(*param_out);
  auto prev_out_eigen = EigenVector<T>::Flatten(*prev_out);
  auto learning_rate_out_eigen = EigenVector<T>::Flatten(*learning_rate_out);

  auto product_eigen = grad_eigen * prev_eigen;
  for (int i = 0; i < product_eigen.size(); i++) {
    if (product_eigen[i] > 0) {
      grad_sign_eigen[i] = 1;
      eta_eigen[i] = eta_positive;
    } else if (product_eigen[i] = 0) {
      grad_sign_eigen[i] = 0;
      eta_eigen[i] = 1;
    } else if (product_eigen[i] < 0) {
      grad_eigen[i] = 0;
      grad_sign_eigen[i] = 0;
      eta_eigen[i] = eta_negative;
    }
  }

  learning_rate_eigen = learning_rate_eigen * eta_eigen;
  for (int i = 0; i < learning_rate_eigen.size(); i++) {
    if (learning_rate_eigen[i] > delta_max) {
      learning_rate_eigen[i] = delta_max;
    } else if (learning_rate_eigen[i] < delta_min) {
      learning_rate_eigen[i] = delta_min;
    }
  }

  param_out_eigen = param_eigen - grad_sign_eigen * learning_rate_eigen;
  prev_out_eigen = grad_eigen;
  learning_rate_out_eigen = learning_rate_eigen;
}

template <typename T, typename Context>
void RpropKernel(const Context& dev_ctx,
                 const DenseTensor& param,
                 const DenseTensor& grad,
                 const DenseTensor& prev,
                 const DenseTensor& learning_rate,
                 const paddle::optional<DenseTensor>& master_param UNUSED,
                 float delta_min,
                 float delta_max,
                 float eta_negative,
                 float eta_positive,
                 bool multi_precision UNUSED,
                 DenseTensor* param_out,
                 DenseTensor* prev_out,
                 DenseTensor* learning_rate_out,
                 DenseTensor* master_param_out UNUSED) {
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(prev_out);
  dev_ctx.template Alloc<T>(learning_rate_out);
  RpropKernelCPUImpl<T>(param,
                        grad,
                        prev,
                        learning_rate,
                        delta_min,
                        delta_max,
                        eta_negative,
                        eta_positive,
                        param_out,
                        prev_out,
                        learning_rate_out);
}

}  // namespace phi

PD_REGISTER_KERNEL(rprop,
                   CPU,
                   ALL_LAYOUT,
                   phi::RpropKernel,
                   phi::dtype::bfloat16,
                   float,
                   double) {}
