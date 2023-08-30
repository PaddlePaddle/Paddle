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

#include "paddle/phi/core/dense_tensor.h"

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
                       DenseTensor* grad_out);

}  // namespace phi
