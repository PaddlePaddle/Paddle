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
#include "paddle/phi/kernels/nadam_kernel.h"

namespace phi {

template <typename T, typename Context>
void NAdamKernel(const Context& dev_ctx,
                 const DenseTensor& param,
                 const DenseTensor& grad,
                 const DenseTensor& learning_rate,
                 const DenseTensor& momentum_decay_pow,
                 const DenseTensor& beta2_pow,
                 const DenseTensor& mu_product,
                 const DenseTensor& moment1,
                 const DenseTensor& moment2,
                 const paddle::optional<DenseTensor>& master_param UNUSED,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float momentum_decay,
                 bool multi_precision UNUSED,
                 DenseTensor* param_out,
                 DenseTensor* momentum_decay_pow_out,
                 DenseTensor* beta2_pow_out,
                 DenseTensor* mu_product_out,
                 DenseTensor* moment1_out,
                 DenseTensor* moment2_out,
                 DenseTensor* master_param_out UNUSED) {
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(momentum_decay_pow_out);
  dev_ctx.template Alloc<T>(beta2_pow_out);
  dev_ctx.template Alloc<T>(mu_product_out);
  dev_ctx.template Alloc<T>(moment1_out);
  dev_ctx.template Alloc<T>(moment2_out);

  T beta1_ = static_cast<T>(beta1);
  T beta2_ = static_cast<T>(beta2);
  T epsilon_ = static_cast<T>(epsilon);
  T momentum_decay_ = static_cast<T>(momentum_decay);

  auto eigen_param = EigenVector<T>::Flatten(param);
  auto eigen_grad = EigenVector<T>::Flatten(grad);
  auto eigen_lr = EigenVector<T>::Flatten(learning_rate);
  auto eigen_momentum_decay_pow = EigenVector<T>::Flatten(momentum_decay_pow);
  auto eigen_beta2_pow = EigenVector<T>::Flatten(beta2_pow);
  auto eigen_mu_product = EigenVector<T>::Flatten(mu_product);
  auto eigen_moment1 = EigenVector<T>::Flatten(moment1);
  auto eigen_moment2 = EigenVector<T>::Flatten(moment2);

  auto eigen_param_out = EigenVector<T>::Flatten(*param_out);
  auto eigen_momentum_decay_pow_out =
      EigenVector<T>::Flatten(*momentum_decay_pow_out);
  auto eigen_beta2_pow_out = EigenVector<T>::Flatten(*beta2_pow_out);
  auto eigen_mu_product_out = EigenVector<T>::Flatten(*mu_product_out);
  auto eigen_moment1_out = EigenVector<T>::Flatten(*moment1_out);
  auto eigen_moment2_out = EigenVector<T>::Flatten(*moment2_out);

  eigen_momentum_decay_pow_out =
      eigen_momentum_decay_pow * static_cast<T>(0.96);
  eigen_beta2_pow_out = eigen_beta2_pow * beta2_;

  auto eigen_mu_t =
      beta1_ *
      (static_cast<T>(1) -
       static_cast<T>(0.5) * eigen_momentum_decay_pow_out.pow(momentum_decay_));
  auto eigen_mu_t_1 =
      beta1_ *
      (static_cast<T>(1) -
       static_cast<T>(0.5) * eigen_momentum_decay_pow_out.pow(momentum_decay_) *
           std::pow(static_cast<T>(0.96), momentum_decay_));

  eigen_mu_product_out = eigen_mu_product * eigen_mu_t;
  auto eigen_mu_product_t_1 = eigen_mu_product_out * eigen_mu_t_1;

  eigen_moment1_out =
      beta1_ * eigen_moment1 + (static_cast<T>(1) - beta1_) * eigen_grad;
  eigen_moment2_out = beta2_ * eigen_moment2 +
                      (static_cast<T>(1) - beta2_) * eigen_grad * eigen_grad;

  Eigen::DSizes<int, 1> p_dsize(param_out->numel());
  auto eigen_moment1_hat = eigen_mu_t_1 * eigen_moment1_out /
                               (static_cast<T>(1) - eigen_mu_product_t_1) +
                           (static_cast<T>(1) - eigen_mu_t) * eigen_grad /
                               (static_cast<T>(1) - eigen_mu_product_out);
  auto eigen_moment2_hat =
      eigen_moment2_out / (static_cast<T>(1) - eigen_beta2_pow_out);

  eigen_param_out = eigen_param - eigen_lr.broadcast(p_dsize) *
                                      eigen_moment1_hat /
                                      (eigen_moment2_hat.sqrt() + epsilon_);
}
}  // namespace phi
