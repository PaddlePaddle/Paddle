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

/**
 * @brief  Fused Attention Kernel.
 * @param ctx                 device context
 * @param x                   The input tensor.
 * @param ln_scale            (optional) Scale is a 1-dimensional tensor of size
 *                            H. Here, H represents the last dimension of its
 * input tensor.
 * @param ln_bias             (optional) Bias is a 1-dimensional tensor of size
 *                            H. Here, H represents the last dimension of its
 * input tensor.
 * @param qkv_weight          The qkv weight tensor.
 * @param qkv_bias            The qkv bias tensor.
 * @param cache_kv            (optional) The cache KV for generation inference.
 * @param src_mask            (optional) The attention mask tensor in fmha.
 * @param out_linear_w        The out_linear weight tensor.
 * @param out_linear_bias     (optional) The out_linear bias tensor.
 * @param ln_scale_2           (optional) Scale is a 1-dimensional tensor of
 * size H. Here, H represents the last dimension of its input tensor.
 * @param ln_bias_2           (optional) Bias is a 1-dimensional tensor of size
 *                            H. Here, H represents the last dimension of its
 * input tensor.
 * @param num_heads           The number head for multi_head_attention.
 * @param transpose_qkv_wb    The qkv_w shape is (h, 3h), do transpose to it.
 * @param pre_layer_norm      if true, the attention op uses pre_layer_norm
 *                            architecture, else, uses post_layer_norm
 * architecture. [default false].
 * @param epsilon             Constant for numerical stability [default 1e-5].
 * @param attn_dropout_rate   Probability of setting units to zero.
 * @param is_test             (bool, default false) Set to true for inference
 *                            only, false " for training. Some layers may run
 * faster when this is true.
 * @param attn_dropout_fix_seed A flag indicating whether to use a fixed seed to
 *                            generate " random mask. NOTE: DO NOT set this flag
 *                            to true in training. Setting this flag to true is
 * only useful in unittest or for debug that always the same output units will
 * be dropped."
 * @param attn_dropout_seed   Dropout random seed.
 * @param attn_dropout_implementation ["downgrade_in_infer"|"upscale_in_train"]
 *                            There are two kinds of ways to implement dropout
 *                            (the mask below is a tensor have the same shape
 *                 with input the value of mask is 0 or 1, the ratio of 0 is
 * dropout_rate)
 *                            1. downgrade_in_infer(default), downgrade the
 *                               outcome at inference time train: out = input *
 * mask inference: out = input * (1.0 - dropout_rate)
 *                            2. upscale_in_train, upscale the outcome at
 *                               training time, do nothing in inference train:
 * out = input * mask / ( 1.0 - dropout_rate ) inference: out = input dropout op
 * can be removed from the program. the program will be efficient
 * @param dropout_rate        Probability of setting units to zero.
 * @param dropout_fix_seed    A flag indicating whether to use a fixed seed to
 *                            generate " random mask. NOTE: DO NOT set this flag
 *                            to true in training. Setting this flag to true is
 * only useful in unittest or for debug that always the same output units will
 * be dropped.
 * @param dropout_seed        Dropout random seed.
 * @param dropout_implementation dropout_implementation
 *                               ["downgrade_in_infer"|"upscale_in_train"] The
 *                               meaning is the same as
 * 'attn_dropout_implementation'
 * @param ln_epsilon          Constant for numerical stability [default 1e-5].
 * @param add_residual        Whether to add residual.
 * @param ring_id             ring id for tensor model parallel. distributed
 * training and inference
 * @param ln_mean             Mean of the current mini batch.
 * @param ln_var              Variance of the current mini batch.
 * @param ln_out              The output tensor after layer_norm.
 * @param qkv_out             Result after qkv.
 * @param qkv_bias_out        Result after qkv and bias op.
 * @param transpose_out_2     Result in fmha.
 * @param qk_out              Result in fmha.
 * @param qktv_out            Result in fmha.
 * @param soft_max_out        Result in fmha.
 * @param attn_dropout_mask_out Result in fmha.
 * @param attn_dropout_out    Result in fmha.
 * @param src_mask_out        Result in fmha.
 * @param fmha_out            Result in fmha.
 * @param out_linear_out      Result after out_linear.
 * @param dropout_mask_out    The random sampled dropout mask.
 * @param ln_mean_2           Mean of the current mini batch.
 * @param ln_var_2            Variance of the current mini batch.
 * @param bias_dropout_residual_out Result of residual + dropout(src + bias).
 * @param cache_kv_out        The update cache KV.
 * @param y                   Result after attention.
 */
template <typename T, typename Context>
void FusedAttentionKernel(const Context &dev_ctx,
                          const DenseTensor &x,
                          const paddle::optional<DenseTensor> &ln_scale,
                          const paddle::optional<DenseTensor> &ln_bias,
                          const DenseTensor &qkv_weight,
                          const paddle::optional<DenseTensor> &qkv_bias,
                          const paddle::optional<DenseTensor> &cache_kv,
                          const paddle::optional<DenseTensor> &src_mask,
                          const DenseTensor &out_linear_weight,
                          const paddle::optional<DenseTensor> &out_linear_bias,
                          const paddle::optional<DenseTensor> &ln_scale_2,
                          const paddle::optional<DenseTensor> &ln_bias_2,
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
                          DenseTensor *ln_mean,
                          DenseTensor *ln_var,
                          DenseTensor *ln_out,
                          DenseTensor *qkv_out,
                          DenseTensor *qkv_bias_out,
                          DenseTensor *transpose_out_2,
                          DenseTensor *qk_out,
                          DenseTensor *qktv_out,
                          DenseTensor *softmax_out,
                          DenseTensor *attn_dropout_mask_out,
                          DenseTensor *attn_dropout_out,
                          DenseTensor *src_mask_out,
                          DenseTensor *fmha_out,
                          DenseTensor *out_linear_out,
                          DenseTensor *dropout_mask_out,
                          DenseTensor *ln_mean_2,
                          DenseTensor *ln_var_2,
                          DenseTensor *bias_dropout_residual_out,
                          DenseTensor *cache_kv_out,
                          DenseTensor *out);

}  // namespace phi
