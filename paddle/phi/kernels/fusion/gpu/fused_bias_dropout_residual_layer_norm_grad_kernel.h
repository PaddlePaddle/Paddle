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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace fusion {
template <typename T, typename Context>
void FusedBiasDropoutResidualLnGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& residual,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& ln_scale,
    const paddle::optional<DenseTensor>& ln_bias,
    const DenseTensor& ln_mean,
    const DenseTensor& ln_variance,
    const DenseTensor& bias_dropout_residual_out,
    const DenseTensor& dropout_mask_out,
    const DenseTensor& y_grad,
    const float dropout_rate,
    const bool is_test,
    const bool dropout_fix_seed,
    const int dropout_seed,
    const std::string& dropout_implementation,
    const float ln_epsilon,
    DenseTensor* x_grad,
    DenseTensor* residual_grad,
    DenseTensor* bias_grad,
    DenseTensor* ln_scale_grad,
    DenseTensor* ln_bias_grad);

}  // namespace fusion
}  // namespace phi
