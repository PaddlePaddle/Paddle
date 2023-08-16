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
void FusedAttentionGradKernel(
    const Context &dev_ctx,
    const DenseTensor &out_grad,
    const DenseTensor &x,
    const DenseTensor &qkv_weight,
    const paddle::optional<DenseTensor> &qkv_bias,
    const paddle::optional<DenseTensor> &qkv_bias_out,
    const paddle::optional<DenseTensor> &src_mask,
    const paddle::optional<DenseTensor> &src_mask_out,
    const DenseTensor &out_linear_weight,
    const paddle::optional<DenseTensor> &out_linear_bias,
    const paddle::optional<DenseTensor> &ln_scale,
    const paddle::optional<DenseTensor> &ln_bias,
    const paddle::optional<DenseTensor> &ln_scale_2,
    const paddle::optional<DenseTensor> &ln_bias_2,
    const paddle::optional<DenseTensor> &ln_out,
    const paddle::optional<DenseTensor> &ln_mean,
    const paddle::optional<DenseTensor> &ln_var,
    const paddle::optional<DenseTensor> &ln_mean_2,
    const paddle::optional<DenseTensor> &ln_var_2,
    const paddle::optional<DenseTensor> &bias_dropout_residual_out,
    const DenseTensor &qkv_out,
    const DenseTensor &transpose_out_2,
    const DenseTensor &qk_out,
    const DenseTensor &qktv_out,
    const DenseTensor &softmax_out,
    const DenseTensor &attn_dropout_mask_out,
    const DenseTensor &attn_dropout_out,
    const DenseTensor &fmha_out,
    const DenseTensor &out_linear_out,
    const DenseTensor &dropout_mask_out,
    int num_heads,
    bool transpose_qkv_wb,
    bool pre_layer_norm,
    float epsilon,
    float attn_dropout_rate,
    bool is_test,
    bool attn_dropout_fix_seed,
    int attn_dropout_seed,
    const std::string &attn_dropout_implementation,
    float dropout_rate,
    bool dropout_fix_seed,
    int dropout_seed,
    const std::string &dropout_implementation,
    float ln_epsilon,
    bool add_residual,
    int ring_id,
    DenseTensor *qkv_bias_grad,
    DenseTensor *qkv_bias_out_grad,
    DenseTensor *src_mask_out_grad,
    DenseTensor *out_linear_bias_grad,
    DenseTensor *ln_scale_grad,
    DenseTensor *ln_bias_grad,
    DenseTensor *ln_scale_2_grad,
    DenseTensor *ln_bias_2_grad,
    DenseTensor *x_grad,
    DenseTensor *qkv_weight_grad,
    DenseTensor *out_linear_weight_grad,
    DenseTensor *ln_out_grad,
    DenseTensor *bias_dropout_residual_out_grad,
    DenseTensor *qkv_out_grad,
    DenseTensor *qktv_out_grad,
    DenseTensor *transpose_out_2_grad,
    DenseTensor *qk_out_grad,
    DenseTensor *softmax_out_grad,
    DenseTensor *attn_dropout_out_grad,
    DenseTensor *fmha_out_grad,
    DenseTensor *out_linear_out_grad);
}  // namespace phi
