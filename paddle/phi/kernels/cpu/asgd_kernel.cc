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
#include "paddle/phi/kernels/funcs/jit/kernels.h"

namespace phi {

template <typename T, typename Context>
void ASGDKernelCPUImpl(const Context& dev_ctx,
                       const DenseTensor& param,
                       const DenseTensor& grad,
                       const DenseTensor& learning_rate,
                       const DenseTensor& d,
                       const DenseTensor& y,
                       const DenseTensor& n,
                       DenseTensor* param_out,
                       DenseTensor* d_out,
                       DenseTensor* y_out) {
  auto param_eigen = EigenVector<T>::Flatten(param);
  auto grad_eigen = EigenVector<T>::Flatten(grad);
  auto d_eigen = EigenVector<T>::Flatten(d);
  auto y_eigen = EigenVector<T>::Flatten(y);
  auto param_out_eigen = EigenVector<T>::Flatten(*param_out);
  auto d_out_eigen = EigenVector<T>::Flatten(*d_out);
  auto y_out_eigen = EigenVector<T>::Flatten(*y_out);
  T learning_rate_T = learning_rate.data<T>()[0];
  T n_T = n.data<T>()[0];

  d_out_eigen = d_eigen - y_eigen + grad_eigen;
  y_out_eigen = grad_eigen;
  param_out_eigen = param_eigen - (learning_rate_T / n_T) * d_out_eigen;
}

template <typename T, typename Context>
void ASGDKernel(const Context& dev_ctx,
                const DenseTensor& param,
                const DenseTensor& grad,
                const DenseTensor& learning_rate,
                const DenseTensor& d,
                const DenseTensor& y,
                const DenseTensor& n,
                const paddle::optional<DenseTensor>& master_param UNUSED,
                bool multi_precision UNUSED,
                DenseTensor* param_out,
                DenseTensor* d_out,
                DenseTensor* y_out,
                DenseTensor* master_param_out UNUSED) {
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(d_out);
  dev_ctx.template Alloc<T>(y_out);
  ASGDKernelCPUImpl<T, Context>(
      dev_ctx, param, grad, learning_rate, d, y, n, param_out, d_out, y_out);
}

}  // namespace phi

PD_REGISTER_KERNEL(asgd, CPU, ALL_LAYOUT, phi::ASGDKernel, float, double) {}
