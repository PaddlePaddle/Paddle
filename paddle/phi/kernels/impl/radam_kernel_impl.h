// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <math.h>

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/radam_kernel.h"

namespace phi {

template <typename T, typename Context>
void RAdamKernel(const Context& dev_ctx,
                 const DenseTensor& param,
                 const DenseTensor& grad,
                 const DenseTensor& learning_rate,
                 const DenseTensor& beta1_pow,
                 const DenseTensor& beta2_pow,
                 const DenseTensor& rho,
                 const DenseTensor& moment1,
                 const DenseTensor& moment2,
                 const paddle::optional<DenseTensor>& master_param UNUSED,
                 float beta1,
                 float beta2,
                 float epsilon,
                 bool multi_precision UNUSED,
                 DenseTensor* param_out,
                 DenseTensor* beta1_pow_out,
                 DenseTensor* beta2_pow_out,
                 DenseTensor* rho_out,
                 DenseTensor* moment1_out,
                 DenseTensor* moment2_out,
                 DenseTensor* master_param_out UNUSED) {
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(beta1_pow_out);
  dev_ctx.template Alloc<T>(beta2_pow_out);
  dev_ctx.template Alloc<T>(rho_out);
  dev_ctx.template Alloc<T>(moment1_out);
  dev_ctx.template Alloc<T>(moment2_out);

  T beta1_ = static_cast<T>(beta1);
  T beta2_ = static_cast<T>(beta2);
  T epsilon_ = static_cast<T>(epsilon);

  auto eigen_param = EigenVector<T>::Flatten(param);
  auto eigen_grad = EigenVector<T>::Flatten(grad);
  auto eigen_lr = EigenVector<T>::Flatten(learning_rate);
  auto eigen_beta1_pow = EigenVector<T>::Flatten(beta1_pow);
  auto eigen_beta2_pow = EigenVector<T>::Flatten(beta2_pow);
  auto eigen_rho = EigenVector<T>::Flatten(rho);
  auto eigen_moment1 = EigenVector<T>::Flatten(moment1);
  auto eigen_moment2 = EigenVector<T>::Flatten(moment2);

  auto eigen_param_out = EigenVector<T>::Flatten(*param_out);
  auto eigen_beta1_pow_out = EigenVector<T>::Flatten(*beta1_pow_out);
  auto eigen_beta2_pow_out = EigenVector<T>::Flatten(*beta2_pow_out);
  auto eigen_rho_out = EigenVector<T>::Flatten(*rho_out);
  auto eigen_moment1_out = EigenVector<T>::Flatten(*moment1_out);
  auto eigen_moment2_out = EigenVector<T>::Flatten(*moment2_out);

  T rho_inf =
      static_cast<T>(2) / (static_cast<T>(1) - beta2_) - static_cast<T>(1);

  eigen_beta1_pow_out = eigen_beta1_pow * beta1_;
  eigen_beta2_pow_out = eigen_beta2_pow * beta2_;
  eigen_rho_out =
      (eigen_rho * (beta2_ - eigen_beta2_pow_out) + eigen_beta2_pow_out) /
      (static_cast<T>(1) - eigen_beta2_pow_out);

  eigen_moment1_out =
      beta1_ * eigen_moment1 + (static_cast<T>(1) - beta1_) * eigen_grad;
  eigen_moment2_out = beta2_ * eigen_moment2 +
                      (static_cast<T>(1) - beta2_) * eigen_grad * eigen_grad;

  Eigen::DSizes<int, 1> p_dsize(param_out->numel());
  auto eigen_moment1_hat =
      eigen_moment1_out / (static_cast<T>(1) - eigen_beta1_pow_out);

  T rho_t = rho_inf - static_cast<T>(2) * eigen_rho_out.data()[0];

  if (rho_t > static_cast<T>(5)) {
    auto l_t = (static_cast<T>(1) - eigen_beta2_pow_out).sqrt() /
               (eigen_moment2_out.sqrt() + epsilon_);
    auto r_t = std::sqrt(
        ((rho_t - static_cast<T>(4)) * (rho_t - static_cast<T>(2)) * rho_inf) /
        ((rho_inf - static_cast<T>(4)) * (rho_inf - static_cast<T>(2)) *
         rho_t));

    eigen_param_out = eigen_param - eigen_lr.broadcast(p_dsize) *
                                        eigen_moment1_hat * r_t * l_t;

  } else {
    eigen_param_out =
        eigen_param - eigen_lr.broadcast(p_dsize) * eigen_moment1_hat;
  }
}
}  // namespace phi
