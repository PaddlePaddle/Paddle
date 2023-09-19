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

#include "paddle/phi/kernels/adamax_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename Context>
void AdamaxKernel(const Context& dev_ctx,
                  const DenseTensor& param,
                  const DenseTensor& grad,
                  const DenseTensor& learning_rate,
                  const DenseTensor& moment,
                  const DenseTensor& inf_norm,
                  const DenseTensor& beta1_pow,
                  const paddle::optional<DenseTensor>& master_param UNUSED,
                  float beta1,
                  float beta2,
                  float epsilon,
                  bool multi_precision UNUSED,
                  DenseTensor* param_out,
                  DenseTensor* moment_out,
                  DenseTensor* inf_norm_out,
                  DenseTensor* master_param_outs UNUSED) {
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(moment_out);
  dev_ctx.template Alloc<T>(inf_norm_out);

  T beta1_ = static_cast<T>(beta1);
  T beta2_ = static_cast<T>(beta2);
  T epsilon_ = static_cast<T>(epsilon);

  auto eigen_param = EigenVector<T>::Flatten(param);
  auto eigen_grad = EigenVector<T>::Flatten(grad);
  auto eigen_moment = EigenVector<T>::Flatten(moment);
  auto eigen_inf_norm = EigenVector<T>::Flatten(inf_norm);
  auto eigen_lr = EigenVector<T>::Flatten(learning_rate);
  auto eigen_beta1_pow = EigenVector<T>::Flatten(beta1_pow);

  auto eigen_param_out = EigenVector<T>::Flatten(*param_out);
  auto eigen_moment_out = EigenVector<T>::Flatten(*moment_out);
  auto eigen_inf_norm_out = EigenVector<T>::Flatten(*inf_norm_out);

  auto& place = *dev_ctx.eigen_device();

  eigen_moment_out.device(place) =
      beta1_ * eigen_moment + (static_cast<T>(1) - beta1_) * eigen_grad;
  eigen_inf_norm_out.device(place) =
      eigen_grad.abs().cwiseMax((beta2_ * eigen_inf_norm) + epsilon_);
  auto lr_t = eigen_lr / (static_cast<T>(1) - eigen_beta1_pow);
  Eigen::DSizes<int, 1> m_dsize(moment_out->numel());
  eigen_param_out.device(place) =
      eigen_param -
      lr_t.broadcast(m_dsize) * (eigen_moment_out / eigen_inf_norm_out);
}

}  // namespace phi
