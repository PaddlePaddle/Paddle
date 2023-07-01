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

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void AdanDenseKernel(const Context& dev_ctx,
                     const DenseTensor& param,
                     const DenseTensor& grad,
                     const DenseTensor& learning_rate,
                     const DenseTensor& pre_grad,
                     const DenseTensor& moment1,
                     const DenseTensor& moment3,
                     const DenseTensor& beta1_pow,
                     const DenseTensor& beta2_pow,
                     const DenseTensor& beta3_pow,
                     const paddle::optional<DenseTensor>& moment2,
                     const paddle::optional<DenseTensor>& master_param,
                     const Scalar& beta1,
                     const Scalar& beta2,
                     const Scalar& beta3,
                     const Scalar& epsilon,
                     const Scalar& weight_decay,
                     bool no_prox,
                     bool multi_precision,
                     bool use_global_beta_pow,
                     bool vanilla,
                     DenseTensor* param_out,
                     DenseTensor* pre_grad_out,
                     DenseTensor* moment1_out,
                     DenseTensor* moment3_out,
                     DenseTensor* beta1_pow_out,
                     DenseTensor* beta2_pow_out,
                     DenseTensor* beta3_pow_out,
                     DenseTensor* moment2_out,
                     DenseTensor* master_param_outs);
}  // namespace phi
