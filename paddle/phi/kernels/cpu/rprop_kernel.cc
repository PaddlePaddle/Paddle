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

template <typename T, typename Context>
void RpropKernelCPUImpl(const Context& dev_ctx,
                        const DenseTensor& param,
                        const DenseTensor& grad,
                        const DenseTensor& prev,
                        const DenseTensor& learning_rate,
                        const DenseTensor& learning_rate_range,
                        const DenseTensor& etas,
                        DenseTensor* param_out,
                        DenseTensor* prev_out,
                        DenseTensor* learning_rate_out) {
  auto param_eigen = EigenVector<T>::Flatten(param);
  auto prev_eigen = EigenVector<T>::Flatten(prev);
  auto param_out_eigen = EigenVector<T>::Flatten(*param_out);
  auto prev_out_eigen = EigenVector<T>::Flatten(*prev_out);
  auto learning_rate_out_eigen = EigenVector<T>::Flatten(*learning_rate_out);
  auto learning_rate_min = learning_rate_range.data<T>()[0];
  auto learning_rate_max = learning_rate_range.data<T>()[1];
  auto eta_negative = etas.data<T>()[0];
  auto eta_positive = etas.data<T>()[1];

  DenseTensor* grad_tensor = new DenseTensor();
  grad_tensor->Resize(grad.dims());
  dev_ctx.template Alloc<T>(grad_tensor);
  phi::Copy<Context>(dev_ctx, grad, dev_ctx.GetPlace(), true, grad_tensor);
  auto grad_eigen = EigenVector<T>::Flatten(*grad_tensor);

  DenseTensor* product_tensor = new DenseTensor();
  product_tensor->Resize(grad.dims());
  dev_ctx.template Alloc<T>(product_tensor);
  auto product_eigen = EigenVector<T>::Flatten(*product_tensor);

  DenseTensor* learning_rate_tensor = new DenseTensor();
  learning_rate_tensor->Resize(learning_rate.dims());
  dev_ctx.template Alloc<T>(learning_rate_tensor);
  phi::Copy<Context>(
      dev_ctx, learning_rate, dev_ctx.GetPlace(), true, learning_rate_tensor);
  auto learning_rate_eigen = EigenVector<T>::Flatten(*learning_rate_tensor);

  DenseTensor* eta_tensor = new DenseTensor();
  eta_tensor->Resize(learning_rate.dims());
  dev_ctx.template Alloc<T>(eta_tensor);
  auto eta_eigen = EigenVector<T>::Flatten(*eta_tensor);

  product_eigen = grad_eigen * prev_eigen;
  T* product_data = product_tensor->data<T>();
  T* grad_data = grad_tensor->data<T>();
  T* eta_data = eta_tensor->data<T>();
  T zero = static_cast<T>(0);
  T one = static_cast<T>(1);
  for (int i = 0, n = product_tensor->numel(); i < n; i++) {
    if (product_data[i] > zero) {
      eta_data[i] = eta_positive;
    } else if (product_data[i] == zero) {
      eta_data[i] = one;
    } else if (product_data[i] < zero) {
      grad_data[i] = zero;
      eta_data[i] = eta_negative;
    }
  }

  learning_rate_eigen = learning_rate_eigen * eta_eigen;
  T* learning_rate_data = learning_rate_tensor->data<T>();
  for (int i = 0, n = learning_rate_tensor->numel(); i < n; i++) {
    if (learning_rate_data[i] > learning_rate_max) {
      learning_rate_data[i] = learning_rate_max;
    } else if (learning_rate_data[i] < learning_rate_min) {
      learning_rate_data[i] = learning_rate_min;
    }
  }

  param_out_eigen = param_eigen - grad_eigen.sign() * learning_rate_eigen;
  prev_out_eigen = grad_eigen;
  learning_rate_out_eigen = learning_rate_eigen;
  phi::Copy<Context>(dev_ctx, *grad_tensor, dev_ctx.GetPlace(), true, prev_out);
  phi::Copy<Context>(dev_ctx,
                     *learning_rate_tensor,
                     dev_ctx.GetPlace(),
                     true,
                     learning_rate_out);
}

template <typename T, typename Context>
void RpropKernel(const Context& dev_ctx,
                 const DenseTensor& param,
                 const DenseTensor& grad,
                 const DenseTensor& prev,
                 const DenseTensor& learning_rate,
                 const paddle::optional<DenseTensor>& master_param UNUSED,
                 const DenseTensor& learning_rate_range,
                 const DenseTensor& etas,
                 bool multi_precision UNUSED,
                 DenseTensor* param_out,
                 DenseTensor* prev_out,
                 DenseTensor* learning_rate_out,
                 DenseTensor* master_param_out UNUSED) {
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(prev_out);
  dev_ctx.template Alloc<T>(learning_rate_out);
  RpropKernelCPUImpl<T, Context>(dev_ctx,
                                 param,
                                 grad,
                                 prev,
                                 learning_rate,
                                 learning_rate_range,
                                 etas,
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
