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
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FusedFeedForwardGradKernel(
    const Context& dev_ctx,
    const DenseTensor& out_grad,
    const DenseTensor& x,
    const DenseTensor& linear1_weight,
    const paddle::optional<DenseTensor>& linear1_bias,
    const DenseTensor& linear2_weight,
    const DenseTensor& dropout1_mask,
    const DenseTensor& dropout2_mask,
    const DenseTensor& linear1_out,
    const DenseTensor& dropout1_out,
    const paddle::optional<DenseTensor>& dropout2_out,
    const paddle::optional<DenseTensor>& ln1_scale,
    const paddle::optional<DenseTensor>& ln1_bias,
    const paddle::optional<DenseTensor>& ln1_out,
    const paddle::optional<DenseTensor>& ln1_mean,
    const paddle::optional<DenseTensor>& ln1_variance,
    const paddle::optional<DenseTensor>& ln2_scale,
    const paddle::optional<DenseTensor>& ln2_bias,
    const paddle::optional<DenseTensor>& ln2_mean,
    const paddle::optional<DenseTensor>& ln2_variance,
    const paddle::optional<DenseTensor>& linear2_bias,
    bool pre_layer_norm,
    float ln1_epsilon,
    float ln2_epsilon,
    const std::string& act_method,
    float dropout1_prob,
    float dropout2_prob,
    const std::string& dropout1_implementation,
    const std::string& dropout2_implementation,
    bool is_test,
    bool dropout1_fix_seed,
    bool dropout2_fix_seed,
    int dropout1_seed_val,
    int dropout2_seed_val,
    bool add_residual,
    int ring_id,
    DenseTensor* x_grad,
    DenseTensor* ln1_scale_grad,
    DenseTensor* ln1_bias_grad,
    DenseTensor* ln2_scale_grad,
    DenseTensor* ln2_bias_grad,
    DenseTensor* linear1_weight_grad,
    DenseTensor* linear1_bias_grad,
    DenseTensor* linear2_weight_grad,
    DenseTensor* linear2_bias_grad);

}  // namespace phi
