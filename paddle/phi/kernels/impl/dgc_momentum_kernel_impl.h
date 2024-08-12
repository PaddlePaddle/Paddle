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

#include "glog/logging.h"

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/momentum_kernel.h"
#include "paddle/phi/kernels/sgd_kernel.h"

namespace phi {

template <typename T, typename Context>
void DGCMomentumKernel(const Context& dev_ctx,
                       const DenseTensor& param,
                       const DenseTensor& grad,
                       const DenseTensor& velocity,
                       const DenseTensor& learning_rate,
                       const DenseTensor& master_param,
                       const DenseTensor& current_step_tensor,
                       const DenseTensor& nranks_tensor,
                       float mu,
                       bool use_nesterov,
                       const std::string& regularization_method,
                       float regularization_coeff,
                       bool multi_precision,
                       float rescale_grad,
                       float rampup_begin_step,
                       DenseTensor* param_out,
                       DenseTensor* velocity_out,
                       DenseTensor* master_param_out,
                       DenseTensor* grad_out) {
  if (static_cast<int>(rampup_begin_step) < 0) {
    return;
  }

  auto* current_step = current_step_tensor.data<T>();

  // nranks
  const int nranks = static_cast<int>(*nranks_tensor.data<float>());
  PADDLE_ENFORCE_GT(
      nranks,
      1,
      common::errors::InvalidArgument(
          "DGC is not useful when num_trainers <= 1, but now nranks=%d",
          nranks));

  auto grad_e = phi::EigenVector<T>::Flatten(grad);
  auto grad_out_e = phi::EigenVector<T>::Flatten(*grad_out);

  auto& eigen_ctx = *dev_ctx.eigen_device();

  // NOTE. In dgc_op we multi grad with nranks, so we need /nranks here.
  grad_out_e.device(eigen_ctx) = (1.0 / nranks) * grad_e;

  VLOG(10) << "current_step:" << *current_step
           << ", rampup_begin_step:" << rampup_begin_step;

  if (static_cast<int>(*current_step) < static_cast<int>(rampup_begin_step)) {
    VLOG(10) << " so use momentum optimizer";

    paddle::optional<phi::DenseTensor> master_param_opt(paddle::none);

    phi::MomentumDenseKernel<T>(dev_ctx,
                                param,
                                grad,
                                velocity,
                                learning_rate,
                                master_param_opt,
                                mu,
                                use_nesterov,
                                regularization_method,
                                regularization_coeff,
                                multi_precision,
                                rescale_grad,
                                param_out,
                                velocity_out,
                                master_param_out);

    return;
  }

  VLOG(10) << " so use sgd optimizer";

  paddle::optional<phi::DenseTensor> master_param_opt(paddle::none);
  if (multi_precision) {
    master_param_opt = master_param;
  }

  phi::SGDDenseKernel<T>(dev_ctx,
                         param,
                         learning_rate,
                         grad,
                         master_param_opt,
                         multi_precision,
                         param_out,
                         master_param_out);
}

}  // namespace phi
