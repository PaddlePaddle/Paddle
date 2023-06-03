/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

// Common InferMeta Functions for multiary operators, The format like:
//
//   1. The number of input MetaTensor is more than 3:
//      void [FunctionDesc|OpName]InferMeta(const MetaTensor& x,
//                                          const MetaTensor& y,
//                                          const MetaTensor& z,
//                                          const MetaTensor& w,
//                                          ...,
//                                          MetaTensor* out) {}
//
//   2. There are `const vector<MetaTensor*>&` in params:
//      void [FunctionDesc|OpName]InferMeta(const vector<MetaTensor*>& x,
//                                          ...,
//                                          MetaTensor* out) {}
//
// NOTE: The InferMeta Functions in this file are arranged in alphabetic order.

std::vector<DDim> GetMetaTensorsDim(
    const std::vector<const MetaTensor*>& tensors);

void AdadeltaInferMeta(const MetaTensor& param,
                       const MetaTensor& grad,
                       const MetaTensor& avg_squared_grad,
                       const MetaTensor& avg_squared_update,
                       const MetaTensor& learning_rate,
                       const MetaTensor& master_param,
                       float rho,
                       float epsilon,
                       bool multi_precision,
                       MetaTensor* param_out,
                       MetaTensor* avg_squared_grad_out,
                       MetaTensor* avg_squared_update_out,
                       MetaTensor* master_param_outs);

void AdagradInferMeta(const MetaTensor& param,
                      const MetaTensor& grad,
                      const MetaTensor& moment,
                      const MetaTensor& learning_rate,
                      const MetaTensor& master_param,
                      float epsilon,
                      bool multi_precision,
                      MetaTensor* param_out,
                      MetaTensor* moment_out,
                      MetaTensor* master_param_out);

void AdamaxInferMeta(const MetaTensor& param,
                     const MetaTensor& grad,
                     const MetaTensor& learning_rate,
                     const MetaTensor& moment,
                     const MetaTensor& inf_norm,
                     const MetaTensor& beta1_pow,
                     const MetaTensor& master_param,
                     float beta1,
                     float beta2,
                     float epsilon,
                     bool multi_precision,
                     MetaTensor* param_out,
                     MetaTensor* moment_out,
                     MetaTensor* inf_norm_out,
                     MetaTensor* master_param_outs);

void AdamInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& moment1,
                   const MetaTensor& moment2,
                   const MetaTensor& beta1_pow,
                   const MetaTensor& beta2_pow,
                   const MetaTensor& master_param,
                   const MetaTensor& skip_update,
                   const Scalar& beta1,
                   const Scalar& beta2,
                   const Scalar& epsilon,
                   bool lazy_mode,
                   int64_t min_row_size_to_use_multithread,
                   bool multi_precision,
                   bool use_global_beta_pow,
                   MetaTensor* param_out,
                   MetaTensor* moment1_out,
                   MetaTensor* moment2_out,
                   MetaTensor* beta1_pow_out,
                   MetaTensor* beta2_pow_out,
                   MetaTensor* master_param_outs);

void AdamwInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& learning_rate,
                    const MetaTensor& moment1,
                    const MetaTensor& moment2,
                    const MetaTensor& beta1_pow,
                    const MetaTensor& beta2_pow,
                    const MetaTensor& master_param,
                    const MetaTensor& skip_update,
                    const Scalar& beta1,
                    const Scalar& beta2,
                    const Scalar& epsilon,
                    float lr_ratio,
                    float coeff,
                    bool with_decay,
                    bool lazy_mode,
                    int64_t min_row_size_to_use_multithread,
                    bool multi_precision,
                    bool use_global_beta_pow,
                    MetaTensor* param_out,
                    MetaTensor* moment1_out,
                    MetaTensor* moment2_out,
                    MetaTensor* beta1_pow_out,
                    MetaTensor* beta2_pow_out,
                    MetaTensor* master_param_outs);

void AddNInferMeta(const std::vector<const MetaTensor*>& x,
                   MetaTensor* out,
                   MetaConfig config = MetaConfig());

void AddNTensorArrayInferMeta(const std::vector<const MetaTensor*>& x,
                              MetaTensor* out,
                              MetaConfig config);

void AucInferMeta(const MetaTensor& input,
                  const MetaTensor& label,
                  const MetaTensor& stat_pos,
                  const MetaTensor& stat_neg,
                  const MetaTensor& ins_tag_weight,
                  const std::string& curve,
                  int num_thresholds,
                  int slide_steps,
                  MetaTensor* auc,
                  MetaTensor* stat_pos_out,
                  MetaTensor* stat_neg_out,
                  MetaConfig config = MetaConfig());

void AverageAccumulatesInferMeta(const MetaTensor& param,
                                 const MetaTensor& in_sum_1,
                                 const MetaTensor& in_sum_2,
                                 const MetaTensor& in_sum_3,
                                 const MetaTensor& in_num_accumulates,
                                 const MetaTensor& in_old_num_accumulates,
                                 const MetaTensor& in_num_updates,
                                 float average_window,
                                 int64_t max_average_window,
                                 int64_t min_average_window,
                                 MetaTensor* out_sum_1,
                                 MetaTensor* out_sum_2,
                                 MetaTensor* out_sum_3,
                                 MetaTensor* out_num_accumulates,
                                 MetaTensor* out_old_num_accumulates,
                                 MetaTensor* out_num_updates);

void BatchNormInferMeta(const MetaTensor& x,
                        const MetaTensor& mean,
                        const MetaTensor& variance,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        bool is_test,
                        float momentum,
                        float epsilon,
                        const std::string& data_layout,
                        bool use_global_stats,
                        bool trainable_statistics,
                        MetaTensor* y,
                        MetaTensor* mean_out,
                        MetaTensor* variance_out,
                        MetaTensor* saved_mean,
                        MetaTensor* saved_variance,
                        MetaTensor* reserve_space,
                        MetaConfig config = MetaConfig());

void BatchNormInferInferMeta(const MetaTensor& x,
                             const MetaTensor& mean,
                             const MetaTensor& variance,
                             const MetaTensor& scale,
                             const MetaTensor& bias,
                             float momentum,
                             float epsilon,
                             const std::string& data_layout,
                             MetaTensor* y,
                             MetaTensor* mean_out,
                             MetaTensor* variance_out,
                             MetaConfig config = MetaConfig());

void BilinearInferMeta(const MetaTensor& x,
                       const MetaTensor& y,
                       const MetaTensor& weight,
                       const MetaTensor& bias,
                       MetaTensor* out,
                       MetaConfig config = MetaConfig());

void BroadcastTensorsInferMeta(const std::vector<const MetaTensor*>& x,
                               std::vector<MetaTensor*> out);

void CheckFiniteAndUnscaleInferMeta(const std::vector<const MetaTensor*>& xs,
                                    const MetaTensor& scale,
                                    std::vector<MetaTensor*> outs,
                                    MetaTensor* found_infinite);

void CoalesceTensorInferMeta(const std::vector<const MetaTensor*>& input,
                             DataType dtype,
                             bool copy_data,
                             bool set_constant,
                             bool persist_output,
                             float constant,
                             bool use_align,
                             int align_size,
                             int size_of_dtype,
                             const std::vector<int64_t>& concated_shapes,
                             const std::vector<int64_t>& concated_ranks,
                             std::vector<MetaTensor*> output,
                             MetaTensor* fused_output,
                             MetaConfig config = MetaConfig());

void CheckMemoryContinueInferMeta(const std::vector<const MetaTensor*>& input,
                                  MetaTensor* output,
                                  std::vector<MetaTensor*> xout,
                                  MetaConfig config = MetaConfig());

void ConcatInferMeta(const std::vector<const MetaTensor*>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());

void DeformableConvInferMeta(const MetaTensor& x,
                             const MetaTensor& offset,
                             const MetaTensor& filter,
                             const MetaTensor& mask,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& dilations,
                             int deformable_groups,
                             int groups,
                             int im2col_step,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());

void EditDistanceInferMeta(const MetaTensor& hyps,
                           const MetaTensor& refs,
                           const MetaTensor& hypslength,
                           const MetaTensor& refslength,
                           bool normalized,
                           MetaTensor* sequencenum,
                           MetaTensor* out);

void FusedLinearParamGradAddInferMeta(const MetaTensor& x,
                                      const MetaTensor& dout,
                                      const MetaTensor& dweight,
                                      const MetaTensor& dbias,
                                      bool multi_precision,
                                      MetaTensor* dweight_out,
                                      MetaTensor* dbias_out);

void FusionGroupInferMeta(const std::vector<const MetaTensor*>& ins,
                          const std::vector<int>& outs_dtype,
                          const std::vector<int>& inputs_dtype,
                          const std::string& func_name,
                          int type,
                          std::vector<MetaTensor*> outs);

void GenerateProposalsV2InferMeta(const MetaTensor& scores,
                                  const MetaTensor& bbox_deltas,
                                  const MetaTensor& im_shape,
                                  const MetaTensor& anchors,
                                  const MetaTensor& variances,
                                  int pre_nms_top_n,
                                  int post_nms_top_n,
                                  float nms_thresh,
                                  float min_size,
                                  float eta,
                                  bool pixel_offset,
                                  MetaTensor* rpn_rois,
                                  MetaTensor* rpn_roi_probs,
                                  MetaTensor* rpn_rois_num);

void GraphReindexInferMeta(const MetaTensor& x,
                           const MetaTensor& neighbors,
                           const MetaTensor& count,
                           const MetaTensor& hashtable_value,
                           const MetaTensor& hashtable_index,
                           MetaTensor* reindex_src,
                           MetaTensor* reindex_dst,
                           MetaTensor* out_nodes);

void GraphSampleNeighborsInferMeta(const MetaTensor& row,
                                   const MetaTensor& col_ptr,
                                   const MetaTensor& x,
                                   const MetaTensor& eids,
                                   const MetaTensor& perm_buffer,
                                   int sample_size,
                                   bool return_eids,
                                   bool flag_perm_buffer,
                                   MetaTensor* out,
                                   MetaTensor* out_count,
                                   MetaTensor* out_eids);

void HSigmoidLossInferMeta(const MetaTensor& x,
                           const MetaTensor& label,
                           const MetaTensor& w,
                           const MetaTensor& bias,
                           const MetaTensor& path,
                           const MetaTensor& code,
                           int num_classes,
                           bool is_sparse,
                           MetaTensor* out,
                           MetaTensor* pre_out,
                           MetaTensor* w_out);

void InterpolateInferMeta(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config = MetaConfig());

void IndexPutInferMeta(const MetaTensor& x,
                       const std::vector<const MetaTensor*>& indices,
                       const MetaTensor& value,
                       bool accumulate,
                       MetaTensor* out);

void LambInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& moment1,
                   const MetaTensor& moment2,
                   const MetaTensor& beta1_pow,
                   const MetaTensor& beta2_pow,
                   const MetaTensor& master_param,
                   const MetaTensor& skip_update,
                   float weight_decay,
                   float beta1,
                   float beta2,
                   float epsilon,
                   bool multi_precision,
                   MetaTensor* param_out,
                   MetaTensor* moment1_out,
                   MetaTensor* moment2_out,
                   MetaTensor* beta1_pow_out,
                   MetaTensor* beta2_pow_out,
                   MetaTensor* master_param_outs);

void LogspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       const MetaTensor& base,
                       DataType dtype,
                       MetaTensor* out);

void MergedAdamInferMeta(
    const std::vector<const MetaTensor*>& param,
    const std::vector<const MetaTensor*>& grad,
    const std::vector<const MetaTensor*>& learning_rate,
    const std::vector<const MetaTensor*>& moment1,
    const std::vector<const MetaTensor*>& moment2,
    const std::vector<const MetaTensor*>& beta1_pow,
    const std::vector<const MetaTensor*>& beta2_pow,
    const paddle::optional<std::vector<const MetaTensor*>>& master_param,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<MetaTensor*> param_out,
    std::vector<MetaTensor*> moment1_out,
    std::vector<MetaTensor*> moment2_out,
    std::vector<MetaTensor*> beta1_pow_out,
    std::vector<MetaTensor*> beta2_pow_out,
    std::vector<MetaTensor*> master_param_out);

void MergedMomentumInferMeta(
    const std::vector<const MetaTensor*>& param,
    const std::vector<const MetaTensor*>& grad,
    const std::vector<const MetaTensor*>& velocity,
    const std::vector<const MetaTensor*>& learning_rate,
    const paddle::optional<std::vector<const MetaTensor*>>& master_param,
    float mu,
    bool use_nesterov,
    const std::vector<std::string>& regularization_method,
    const std::vector<float>& regularization_coeff,
    bool multi_precision,
    float rescale_grad,
    std::vector<MetaTensor*> param_out,
    std::vector<MetaTensor*> velocity_out,
    std::vector<MetaTensor*> master_param_out);

void MemoryEfficientAttentionInferMeta(const MetaTensor& query,
                                       const MetaTensor& key,
                                       const MetaTensor& value,
                                       const MetaTensor& bias,
                                       const MetaTensor& cu_seqlens_q,
                                       const MetaTensor& cu_seqlens_k,
                                       const MetaTensor& causal_diagonal,
                                       const MetaTensor& seqlen_k,
                                       const Scalar& max_seqlen_q,
                                       const Scalar& max_seqlen_k,
                                       const bool causal,
                                       const double dropout_p,
                                       const float scale,
                                       const bool is_test,
                                       MetaTensor* output,
                                       MetaTensor* logsumexp,
                                       MetaTensor* seed_and_offset);

void MeshgridInferMeta(const std::vector<const MetaTensor*>& inputs,
                       std::vector<MetaTensor*> outputs);

void MomentumInferMeta(const MetaTensor& param,
                       const MetaTensor& grad,
                       const MetaTensor& velocity,
                       const MetaTensor& learning_rate,
                       const MetaTensor& master_param,
                       float mu,
                       bool use_nesterov,
                       const std::string& regularization_method,
                       float regularization_coeff,
                       bool multi_precision,
                       float rescale_grad,
                       MetaTensor* param_out,
                       MetaTensor* velocity_out,
                       MetaTensor* master_param_out);

void MultiDotInferMeta(const std::vector<const MetaTensor*>& x,
                       MetaTensor* out);

void MultiplexInferMeta(const std::vector<const MetaTensor*>& ins,
                        const MetaTensor& ids,
                        MetaTensor* out);

void PsroiPoolInferMeta(const MetaTensor& x,
                        const MetaTensor& rois,
                        const MetaTensor& rois_num,
                        int pooled_height,
                        int pooled_width,
                        int output_channels,
                        float spatial_scale,
                        MetaTensor* out);

void RmspropInferMeta(const MetaTensor& param,
                      const MetaTensor& mean_square,
                      const MetaTensor& grad,
                      const MetaTensor& moment,
                      const MetaTensor& learning_rate,
                      const MetaTensor& mean_grad,
                      const MetaTensor& master_param,
                      float epsilon,
                      float decay,
                      float momentum,
                      bool centered,
                      bool multi_precision,
                      MetaTensor* param_out,
                      MetaTensor* moment_out,
                      MetaTensor* mean_square_out,
                      MetaTensor* mean_grad_out,
                      MetaTensor* master_param_outs);

void RnnInferMeta(const MetaTensor& x,
                  const std::vector<const MetaTensor*>& pre_state,
                  const std::vector<const MetaTensor*>& weight_list,
                  const MetaTensor& sequence_length,
                  float dropout_prob,
                  bool is_bidirec,
                  int input_size,
                  int hidden_size,
                  int num_layers,
                  const std::string& mode,
                  int seed,
                  bool is_test,
                  MetaTensor* out,
                  MetaTensor* dropout_state,
                  std::vector<MetaTensor*> state,
                  MetaTensor* reserve);

void SendUERecvInferMeta(const MetaTensor& x,
                         const MetaTensor& y,
                         const MetaTensor& src_index,
                         const MetaTensor& dst_index,
                         const std::string& message_op,
                         const std::string& reduce_op,
                         const IntArray& out_size,
                         MetaTensor* out,
                         MetaTensor* dst_count);

void SendUVInferMeta(const MetaTensor& x,
                     const MetaTensor& y,
                     const MetaTensor& src_index,
                     const MetaTensor& dst_index,
                     const std::string& message_op,
                     MetaTensor* out);

void SgdInferMeta(const MetaTensor& param,
                  const MetaTensor& learning_rate,
                  const MetaTensor& grad,
                  const MetaTensor& master_param,
                  bool multi_precision,
                  MetaTensor* param_out,
                  MetaTensor* master_param_out);

void StackInferMeta(const std::vector<const MetaTensor*>& x,
                    int axis,
                    MetaTensor* out,
                    MetaConfig config = MetaConfig());

void UnchangedMultiInferMeta(const std::vector<const MetaTensor*>& x,
                             std::vector<MetaTensor*> out);

void ShareBufferInferMeta(const std::vector<const MetaTensor*>& x,
                          const std::vector<bool>& share_dims_and_dtype,
                          std::vector<MetaTensor*> out,
                          std::vector<MetaTensor*> xout);

void UpdateLossScalingInferMeta(const std::vector<const MetaTensor*>& xs,
                                const MetaTensor& found_infinite,
                                const MetaTensor& prev_loss_scaling,
                                const MetaTensor& in_good_steps,
                                const MetaTensor& in_bad_steps,
                                std::vector<MetaTensor*> outs,
                                MetaTensor* loss_scaling,
                                MetaTensor* out_good_steps,
                                MetaTensor* out_bad_steps);

void WarpctcInferMeta(const MetaTensor& logits,
                      const MetaTensor& label,
                      const MetaTensor& logits_length,
                      const MetaTensor& labels_length,
                      int blank,
                      bool norm_by_times,
                      MetaTensor* loss,
                      MetaTensor* warpctcgrad);

void WarprnntInferMeta(const MetaTensor& input,
                       const MetaTensor& label,
                       const MetaTensor& input_lengths,
                       const MetaTensor& label_lengths,
                       int blank,
                       float fastemit_lambda,
                       MetaTensor* loss,
                       MetaTensor* warpctcgrad);

void WeightedSampleNeighborsInferMeta(const MetaTensor& row,
                                      const MetaTensor& col_ptr,
                                      const MetaTensor& edge_weight,
                                      const MetaTensor& x,
                                      const MetaTensor& eids,
                                      int sample_size,
                                      bool return_eids,
                                      MetaTensor* out,
                                      MetaTensor* out_count,
                                      MetaTensor* out_eids);

void WhereInferMeta(const MetaTensor& condition,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    MetaTensor* out);

void YoloLossInferMeta(const MetaTensor& x,
                       const MetaTensor& gt_box,
                       const MetaTensor& gt_label,
                       const MetaTensor& gt_score,
                       const std::vector<int>& anchors,
                       const std::vector<int>& anchor_mask,
                       int class_num,
                       float ignore_thresh,
                       int downsample_ratio,
                       bool use_label_smooth,
                       float scale_x_y,
                       MetaTensor* loss,
                       MetaTensor* objectness_mask,
                       MetaTensor* gt_match_mask);

void FusedAdamInferMeta(
    const std::vector<const MetaTensor*>& params,
    const std::vector<const MetaTensor*>& grads,
    const MetaTensor& learning_rate,
    const std::vector<const MetaTensor*>& moments1,
    const std::vector<const MetaTensor*>& moments2,
    const std::vector<const MetaTensor*>& beta1_pows,
    const std::vector<const MetaTensor*>& beta2_pows,
    const paddle::optional<std::vector<const MetaTensor*>>& master_params,
    const MetaTensor& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    int chunk_size,
    float weight_decay,
    bool use_adamw,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<MetaTensor*> params_out,
    std::vector<MetaTensor*> moments1_out,
    std::vector<MetaTensor*> moments2_out,
    std::vector<MetaTensor*> beta1_pows_out,
    std::vector<MetaTensor*> beta2_pows_out,
    std::vector<MetaTensor*> master_params_out);

void FusedConvInferMeta(const MetaTensor& input,
                        const MetaTensor& filter,
                        const MetaTensor& bias,
                        const MetaTensor& residual_param,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::string& padding_algorithm,
                        const std::vector<int>& dilations,
                        int groups,
                        const std::string& data_format,
                        const std::string& mkldnn_data_type,
                        const std::string& fuse_activation,
                        bool fuse_residual_conn,
                        bool force_fp32_output,
                        MetaTensor* out,
                        MetaConfig config);

void MoeInferMeta(const MetaTensor& x,
                  const MetaTensor& gate,
                  const MetaTensor& bmm0,
                  const MetaTensor& bias0,
                  const MetaTensor& bmm1,
                  const MetaTensor& bias1,
                  const std::string& act_type,
                  MetaTensor* out);

}  // namespace phi
