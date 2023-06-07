/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/meta_tensor.h"

namespace phi {

// Common InferMeta Functions for fusion operators.
// NOTE: The InferMeta Functions in this file are arranged in alphabetic order.

void AddActXPUInferMeta(const MetaTensor& x,
                        const MetaTensor& x_max,
                        const MetaTensor& y,
                        const MetaTensor& y_max,
                        int act_type,
                        MetaTensor* out,
                        MetaTensor* out_max);

void Conv2dXPUInferMeta(const MetaTensor& x,
                        const MetaTensor& x_max,
                        const MetaTensor& filter,
                        const MetaTensor& filter_max,
                        const MetaTensor& bias,
                        const MetaTensor& branch,
                        const MetaTensor& branch_max,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations,
                        const std::vector<int>& strides,
                        const std::string& padding_algorithm,
                        int groups,
                        bool has_bias,
                        bool has_branch,
                        int act_type,
                        float act_param,
                        MetaTensor* out,
                        MetaTensor* out_max);

void EmbeddingWithEltwiseAddXPUInferMeta(
    const std::vector<const MetaTensor*>& ids,
    const std::vector<const MetaTensor*>& tables,
    const MetaTensor& mask,
    MetaTensor* out,
    MetaTensor* seq_lod,
    MetaTensor* max_seq_len);

void FcXPUInferMeta(const MetaTensor& x,
                    const MetaTensor& x_max,
                    const MetaTensor& w,
                    const MetaTensor& w_max,
                    const MetaTensor& bias,
                    int in_num_col_dims,
                    bool transpose_x,
                    float alpha,
                    float beta,
                    int act_type,
                    float act_alpha,
                    MetaTensor* out,
                    MetaTensor* out_max);

void GenerateSequenceXPUInferMeta(const MetaTensor& x,
                                  DataType dtype,
                                  MetaTensor* out);

void MultiEncoderXPUInferMeta(
    const MetaTensor& x,
    const std::vector<const MetaTensor*>& fc_weight,
    const std::vector<const MetaTensor*>& fc_weight_max,
    const std::vector<const MetaTensor*>& fc_bias,
    const std::vector<const MetaTensor*>& ln_scale,
    const std::vector<const MetaTensor*>& ln_bias,
    const MetaTensor& mask,
    const MetaTensor& seq_lod,
    const MetaTensor& max_seq_len,
    int layer_num,
    bool norm_before,
    int hidden_dim,
    int head_num,
    int size_per_head,
    int ffn_hidden_dim_scale,
    int act_type,
    int relative_type,
    int slice_idx,
    MetaTensor* out,
    MetaTensor* x_fp16,
    MetaTensor* out_fp16);

void FusedMultiTransformerXpuInferMeta(
    const MetaTensor& x,
    const std::vector<const MetaTensor*>& ln_scale,
    const std::vector<const MetaTensor*>& ln_bias,
    const std::vector<const MetaTensor*>& qkvw,
    const std::vector<const MetaTensor*>& qkvw_max,
    const std::vector<const MetaTensor*>& qkv_bias,
    const std::vector<const MetaTensor*>& out_linear_w,
    const std::vector<const MetaTensor*>& out_linear_wmax,
    const std::vector<const MetaTensor*>& out_linear_bias,
    const std::vector<const MetaTensor*>& ffn_ln_scale,
    const std::vector<const MetaTensor*>& ffn_ln_bias,
    const std::vector<const MetaTensor*>& ffn1_weight,
    const std::vector<const MetaTensor*>& ffn1_weight_max,
    const std::vector<const MetaTensor*>& ffn1_bias,
    const std::vector<const MetaTensor*>& ffn2_weight,
    const std::vector<const MetaTensor*>& ffn2_weight_max,
    const std::vector<const MetaTensor*>& ffn2_bias,
    const std::vector<const MetaTensor*>& cache_kv,
    const std::vector<const MetaTensor*>& pre_caches,
    const std::vector<const MetaTensor*>& rotary_pos_emb,
    const std::vector<const MetaTensor*>& time_step,
    const std::vector<const MetaTensor*>& seq_lengths,
    const std::vector<const MetaTensor*>& src_mask,
    const std::vector<const MetaTensor*>& gather_index,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    int ring_id,
    int gather_axis,
    MetaTensor* out,
    std::vector<MetaTensor*> cache_kv_out);

void YoloBoxXPUInferMeta(const MetaTensor& x,
                         const MetaTensor& x_max,
                         const MetaTensor& grid,
                         const MetaTensor& stride,
                         const MetaTensor& anchor_grid,
                         float offset,
                         MetaTensor* out,
                         MetaTensor* out_max);

}  // namespace phi
