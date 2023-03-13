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
void FusionAttentionGradKernel(const Context& ctx,
                               const DenseTensor& x,
                               const DenseTensor& ln_mean,
                               const DenseTensor& ln_variance,
                               const DenseTensor& ln_out_grad,
                               const DenseTensor& qkv_out_grad,
                               const DenseTensor& qkv_bias_out_grad,
                               const DenseTensor& transpose_out2_grad,
                               const DenseTensor& qk_out_grad,
                               const DenseTensor& qkv_out2,
                               const DenseTensor& soft_max_out_grad,
                               const DenseTensor& attn_dropout_mask_out_grad,
                               const DenseTensor& attn_dropout_out,
                               const DenseTensor& src_mask_out_grad,
                               const DenseTensor& fmha_out_grad,
                               const DenseTensor& out_linear_out_grad,
                               const DenseTensor& dropout_mask_out,
                               const DenseTensor& ln2mean,
                               const DenseTensor& ln2variance,
                               const DenseTensor& bias_dropout_residual_out_grad,
                               const DenseTensor& cache_kv_out,
                               const DenseTensor& y,
                               int num_heads,
                               bool transpose_qkv_wb,
                               bool pre_layer_norm,
                               float epsilon,
                               float attn_dropout_rate,
                               bool is_test,
                               bool attn_dropout_fix_seed,
                               int attn_dropout_seed,
                               std::string attn_dropout_implementation,
                               float dropout_rate,
                               bool dropout_fix_seed,
                               int dropout_seed,
                               std::string dropout_implementation,
                               float ln2_epsilon,
                               bool add_residual,
                               int ring_id,
                               DenseTensor* x_grad,
                               DenseTensor* ln_scale_grad,
                               DenseTensor* ln_bias_grad,
                               DenseTensor* qkvw_grad,
                               DenseTensor* qkv_bias_grad,
                               DenseTensor* cache_kv,
                               DenseTensor* src_mask,
                               DenseTensor* out_linear_w_grad,
                               DenseTensor* out_linear_bias_grad,
                               DenseTensor* ln2scale_grad,
                               DenseTensor* ln2bias_grad);
}  // namespace phi
