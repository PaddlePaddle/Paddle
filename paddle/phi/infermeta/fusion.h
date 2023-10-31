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

void AddLayernormXPUInferMeta(const MetaTensor& x,
                              const MetaTensor& y,
                              const MetaTensor& scale,
                              const MetaTensor& bias,
                              int begin_norm_axis,
                              float epsilon,
                              MetaTensor* out);

void Conv1dXPUInferMeta(const MetaTensor& x,
                        const MetaTensor& x_max,
                        const MetaTensor& filter,
                        const MetaTensor& filter_max,
                        const MetaTensor& bias,
                        const MetaTensor& branch,
                        const MetaTensor& branch_max,
                        const std::vector<int>& paddings,
                        const std::string& padding_algorithm,
                        int dilations,
                        int strides,
                        int groups,
                        int act_type,
                        float act_param,
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
                        int act_type,
                        float act_param,
                        DataType out_dtype,
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
                    DataType out_dtype,
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

void FusedAttentionInferMeta(const MetaTensor& x,
                             const MetaTensor& ln_scale,
                             const MetaTensor& ln_bias,
                             const MetaTensor& qkv_weight,
                             const MetaTensor& qkv_bias,
                             const MetaTensor& cache_kv,
                             const MetaTensor& src_mask,
                             const MetaTensor& out_linear_weight,
                             const MetaTensor& out_linear_bias,
                             const MetaTensor& ln_scale_2,
                             const MetaTensor& ln_bias_2,
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
                             MetaTensor* ln_mean,
                             MetaTensor* ln_var,
                             MetaTensor* ln_out,
                             MetaTensor* qkv_out,
                             MetaTensor* qkv_bias_out,
                             MetaTensor* transpose_out_2,
                             MetaTensor* qk_out,
                             MetaTensor* qktv_out,
                             MetaTensor* softmax_out,
                             MetaTensor* attn_dropout_mask_out,
                             MetaTensor* attn_dropout_out,
                             MetaTensor* src_mask_out,
                             MetaTensor* fmha_out,
                             MetaTensor* out_linear_out,
                             MetaTensor* dropout_mask_out,
                             MetaTensor* ln_mean_2,
                             MetaTensor* ln_var_2,
                             MetaTensor* bias_dropout_residual_out,
                             MetaTensor* cache_kv_out,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());

void FusedAttentionGradInferMeta(const MetaTensor& out_grad,
                                 const MetaTensor& x,
                                 const MetaTensor& qkv_weight,
                                 const MetaTensor& qkv_bias,
                                 const MetaTensor& qkv_bias_out,
                                 const MetaTensor& src_mask,
                                 const MetaTensor& src_mask_out,
                                 const MetaTensor& out_linear_weight,
                                 const MetaTensor& out_linear_bias,
                                 const MetaTensor& ln_scale,
                                 const MetaTensor& ln_bias,
                                 const MetaTensor& ln_scale_2,
                                 const MetaTensor& ln_bias_2,
                                 const MetaTensor& ln_out,
                                 const MetaTensor& ln_mean,
                                 const MetaTensor& ln_var,
                                 const MetaTensor& ln_mean_2,
                                 const MetaTensor& ln_var_2,
                                 const MetaTensor& bias_dropout_residual_out,
                                 const MetaTensor& qkv_out,
                                 const MetaTensor& transpose_out_2,
                                 const MetaTensor& qk_out,
                                 const MetaTensor& qktv_out,
                                 const MetaTensor& softmax_out,
                                 const MetaTensor& attn_dropout_mask_out,
                                 const MetaTensor& attn_dropout_out,
                                 const MetaTensor& fmha_out,
                                 const MetaTensor& out_linear_out,
                                 const MetaTensor& dropout_mask_out,
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
                                 MetaTensor* qkv_bias_grad,
                                 MetaTensor* qkv_bias_out_grad,
                                 MetaTensor* src_mask_out_grad,
                                 MetaTensor* out_linear_bias_grad,
                                 MetaTensor* ln_scale_grad,
                                 MetaTensor* ln_bias_grad,
                                 MetaTensor* ln_scale_2_grad,
                                 MetaTensor* ln_bias_2_grad,
                                 MetaTensor* x_grad,
                                 MetaTensor* qkv_weight_grad,
                                 MetaTensor* out_linear_weight_grad,
                                 MetaTensor* ln_out_grad,
                                 MetaTensor* bias_dropout_residual_out_grad,
                                 MetaTensor* qkv_out_grad,
                                 MetaTensor* qktv_out_grad,
                                 MetaTensor* transpose_out_2_grad,
                                 MetaTensor* qk_out_grad,
                                 MetaTensor* softmax_out_grad,
                                 MetaTensor* attn_dropout_out_grad,
                                 MetaTensor* fmha_out_grad,
                                 MetaTensor* out_linear_out_grad);

void FusedFeedForwardInferMeta(const MetaTensor& x,
                               const MetaTensor& dropout1_seed,
                               const MetaTensor& dropout2_seed,
                               const MetaTensor& linear1_weight,
                               const MetaTensor& linear1_bias,
                               const MetaTensor& linear2_weight,
                               const MetaTensor& linear2_bias,
                               const MetaTensor& ln1_scale,
                               const MetaTensor& ln1_bias,
                               const MetaTensor& ln2_scale,
                               const MetaTensor& ln2_bias,
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
                               MetaTensor* out,
                               MetaTensor* dropout1_mask,
                               MetaTensor* dropout2_mask,
                               MetaTensor* ln1_mean,
                               MetaTensor* ln1_variance,
                               MetaTensor* ln2_mean,
                               MetaTensor* ln2_variance,
                               MetaTensor* linear1_out,
                               MetaTensor* ln1_out,
                               MetaTensor* dropout1_out,
                               MetaTensor* dropout2_out);

void FusedFeedForwardGradInferMeta(const MetaTensor& out_grad,
                                   const MetaTensor& x,
                                   const MetaTensor& linear1_weight,
                                   const MetaTensor& linear1_bias,
                                   const MetaTensor& linear2_weight,
                                   const MetaTensor& dropout1_mask,
                                   const MetaTensor& dropout2_mask,
                                   const MetaTensor& linear1_out,
                                   const MetaTensor& dropout1_out,
                                   const MetaTensor& dropout2_out,
                                   const MetaTensor& ln1_scale,
                                   const MetaTensor& ln1_bias,
                                   const MetaTensor& ln1_out,
                                   const MetaTensor& ln1_mean,
                                   const MetaTensor& ln1_variance,
                                   const MetaTensor& ln2_scale,
                                   const MetaTensor& ln2_bias,
                                   const MetaTensor& ln2_mean,
                                   const MetaTensor& ln2_variance,
                                   const MetaTensor& linear2_bias,
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
                                   MetaTensor* x_grad,
                                   MetaTensor* ln1_scale_grad,
                                   MetaTensor* ln1_bias_grad,
                                   MetaTensor* ln2_scale_grad,
                                   MetaTensor* ln2_bias_grad,
                                   MetaTensor* linear1_weight_grad,
                                   MetaTensor* linear1_bias_grad,
                                   MetaTensor* linear2_weight_grad,
                                   MetaTensor* linear2_bias_grad);

void FusedGemmEpilogueInferMeta(const MetaTensor& x,
                                const MetaTensor& y,
                                const MetaTensor& bias,
                                bool trans_x,
                                bool trans_y,
                                const std::string& activation,
                                MetaTensor* out,
                                MetaTensor* reserve_space);

void FusedGemmEpilogueGradInferMeta(const MetaTensor& x,
                                    const MetaTensor& y,
                                    const MetaTensor& reserve_space,
                                    const MetaTensor& out_grad,
                                    bool trans_x,
                                    bool trans_y,
                                    const std::string& activation_grad,
                                    MetaTensor* x_grad,
                                    MetaTensor* y_grad,
                                    MetaTensor* bias_grad);

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
    const MetaTensor& rotary_pos_emb,
    const MetaTensor& time_step,
    const MetaTensor& seq_lengths,
    const MetaTensor& src_mask,
    const MetaTensor& gather_index,
    const MetaTensor& max_buffer,
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

void Conv2dTransposeXPUInferMeta(const MetaTensor& x,
                                 const MetaTensor& x_max,
                                 const MetaTensor& filter,
                                 const MetaTensor& filter_max,
                                 const MetaTensor& bias,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 const std::vector<int>& output_padding,
                                 const IntArray& output_size,
                                 const std::string& padding_algorithm,
                                 int groups,
                                 const std::vector<int>& dilations,
                                 const std::string& data_format,
                                 bool has_bias,
                                 bool with_act,
                                 const std::string& act_type,
                                 MetaTensor* out,
                                 MetaTensor* out_max);

void FastWhereXPUInferMeta(const MetaTensor& condition,
                           const MetaTensor& x,
                           const MetaTensor& y,
                           MetaTensor* out);

void FastLayernormXPUInferMeta(const MetaTensor& x,
                               const MetaTensor& scale,
                               const MetaTensor& bias,
                               int begin_norm_axis,
                               float epsilon,
                               MetaTensor* out);

void BNActXPUInferMeta(const MetaTensor& x,
                       const MetaTensor& mean,
                       const MetaTensor& variance,
                       const MetaTensor& scale,
                       const MetaTensor& bias,
                       float momentum,
                       float epsilon,
                       const std::string& data_layout,
                       int act_type,
                       MetaTensor* y,
                       MetaConfig config = MetaConfig());

void AddCMulXPUInferMeta(const MetaTensor& x,
                         const MetaTensor& y,
                         const MetaTensor& w,
                         MetaTensor* out);

void LayerNormActXPUInferMeta(const MetaTensor& x,
                              const MetaTensor& scale,
                              const MetaTensor& bias,
                              int begin_norm_axis,
                              float epsilon,
                              int act_type,
                              float act_param,
                              MetaTensor* y);

void FusedScaleBiasReluConvBnstatsInferMeta(
    const MetaTensor& x,
    const MetaTensor& w,
    const MetaTensor& scale,
    const MetaTensor& bias,
    const MetaTensor& bn_scale,
    const MetaTensor& bn_bias,
    const MetaTensor& input_running_mean,
    const MetaTensor& input_running_var,
    const std::vector<int>& paddings,
    const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const std::string& padding_algorithm,
    int groups,
    const std::string& data_format,
    float momentum,
    float epsilon,
    bool fuse_prologue,
    bool exhaustive_search,
    int64_t accumulation_count,
    MetaTensor* out,
    MetaTensor* out_running_mean,
    MetaTensor* out_running_var,
    MetaTensor* saved_mean,
    MetaTensor* saved_var,
    MetaTensor* eq_scale,
    MetaTensor* eq_bias);

void SqueezeExcitationInferMeta(const MetaTensor& x,
                                const MetaTensor& filter,
                                const MetaTensor& filter_max,
                                const MetaTensor& bias,
                                const MetaTensor& branch,
                                const std::vector<int>& act_type,
                                const std::vector<float>& act_param,
                                const std::vector<int>& filter_dims,
                                MetaTensor* out);

void FusedEmbeddingEltWiseLayerNormInferMeta(
    const std::vector<const MetaTensor*>& ids,
    const std::vector<const MetaTensor*>& embs,
    const MetaTensor& bias,
    const MetaTensor& scale,
    const float epsilon,
    MetaTensor* out);

void FusionTransposeFlattenConcatInferMeta(
    const std::vector<const MetaTensor*>& x,
    const std::vector<int>& trans_axis,
    const int flatten_axis,
    const int concat_axis,
    MetaTensor* out);

void FusedFCElementwiseLayerNormInferMeta(const MetaTensor& x,
                                          const MetaTensor& w,
                                          const MetaTensor& y,
                                          const MetaTensor& bias0,
                                          const MetaTensor& scale,
                                          const MetaTensor& bias1,
                                          const int x_num_col_dims,
                                          const std::string& activation_type,
                                          const float epsilon,
                                          const int begin_norm_axis,
                                          MetaTensor* out,
                                          MetaTensor* mean,
                                          MetaTensor* variance,
                                          MetaConfig config = MetaConfig());

void Conv2dFusionInferMeta(const MetaTensor& input,
                           const MetaTensor& filter,
                           const MetaTensor& bias,
                           const MetaTensor& residual_data,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::string& padding_algorithm,
                           const std::vector<int>& dilations,
                           int groups,
                           const std::string& data_format,
                           const std::string& activation,
                           bool exhaustive_search,
                           const std::vector<int>& split_channels,
                           int workspace_size_MB,
                           MetaTensor* output,
                           std::vector<MetaTensor*> outputs);

}  // namespace phi
