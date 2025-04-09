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

#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace primitive {

using IntArray = paddle::experimental::IntArray;
std::vector<std::vector<paddle::Tensor>> add_n_vjp(
    const std::vector<paddle::Tensor>& x,
    const Tensor& out_grad,
    const std::vector<std::vector<bool>>& stop_gradients);

std::vector<std::vector<paddle::Tensor>> reshape_vjp(
    const Tensor& xshape,
    const Tensor& out_grad,
    const std::vector<std::vector<bool>>& stop_gradients);

std::vector<std::vector<paddle::Tensor>> fused_attention_vjp(
    const Tensor& out_grad,
    const Tensor& x,
    const Tensor& qkv_weight,
    const paddle::optional<Tensor>& qkv_bias,
    const paddle::optional<Tensor>& qkv_bias_out,
    const paddle::optional<Tensor>& src_mask,
    const paddle::optional<Tensor>& src_mask_out,
    const Tensor& out_linear_weight,
    const paddle::optional<Tensor>& out_linear_bias,
    const paddle::optional<Tensor>& ln_scale,
    const paddle::optional<Tensor>& ln_bias,
    const paddle::optional<Tensor>& ln_scale_2,
    const paddle::optional<Tensor>& ln_bias_2,
    const paddle::optional<Tensor>& ln_out,
    const paddle::optional<Tensor>& ln_mean,
    const paddle::optional<Tensor>& ln_var,
    const paddle::optional<Tensor>& ln_mean_2,
    const paddle::optional<Tensor>& ln_var_2,
    const paddle::optional<Tensor>& bias_dropout_residual_out,
    const Tensor& qkv_out,
    const Tensor& transpose_out_2,
    const Tensor& qk_out,
    const Tensor& qktv_out,
    const Tensor& softmax_out,
    const Tensor& attn_dropout_mask_out,
    const Tensor& attn_dropout_out,
    const Tensor& fmha_out,
    const Tensor& out_linear_out,
    const Tensor& dropout_mask_out,
    int num_heads,
    bool transpose_qkv_wb,
    bool pre_layer_norm,
    float epsilon,
    float attn_dropout_rate,
    bool is_test,
    bool attn_dropout_fix_seed,
    int attn_dropout_seed,
    const std::string& attn_dropout_implementation,
    float dropout_rate,
    bool dropout_fix_seed,
    int dropout_seed,
    const std::string& dropout_implementation,
    float ln_epsilon,
    bool add_residual,
    int ring_id,
    const std::vector<std::vector<bool>>& stop_gradients);

std::vector<std::vector<paddle::Tensor>> fused_gemm_epilogue_vjp(
    const Tensor& x,
    const Tensor& y,
    const paddle::optional<Tensor>& reserve_space,
    const Tensor& out_grad,
    bool trans_x,
    bool trans_y,
    const std::string& activation,
    const std::vector<std::vector<bool>>& stop_gradients);

}  // namespace primitive
}  // namespace paddle
