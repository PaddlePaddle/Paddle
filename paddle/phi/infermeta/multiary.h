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
                       float rho,
                       float epsilon,
                       MetaTensor* param_out,
                       MetaTensor* avg_squared_grad_out,
                       MetaTensor* avg_squared_update_out);

void AdagradInferMeta(const MetaTensor& param,
                      const MetaTensor& grad,
                      const MetaTensor& moment,
                      const MetaTensor& learning_rate,
                      float epsilon,
                      MetaTensor* param_out,
                      MetaTensor* moment_out);

void AdamaxInferMeta(const MetaTensor& param,
                     const MetaTensor& grad,
                     const MetaTensor& learning_rate,
                     const MetaTensor& moment,
                     const MetaTensor& inf_norm,
                     const MetaTensor& beta1_pow,
                     float beta1,
                     float beta2,
                     float epsilon,
                     MetaTensor* param_out,
                     MetaTensor* moment_out,
                     MetaTensor* inf_norm_out);

void AdamInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& moment1,
                   const MetaTensor& moment2,
                   const MetaTensor& beta1_pow,
                   const MetaTensor& beta2_pow,
                   paddle::optional<const MetaTensor&> master_param,
                   paddle::optional<const MetaTensor&> skip_update,
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
                    paddle::optional<const MetaTensor&> master_param,
                    paddle::optional<const MetaTensor&> skip_update,
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

void AucInferMeta(const MetaTensor& input,
                  const MetaTensor& label,
                  const MetaTensor& stat_pos,
                  const MetaTensor& stat_neg,
                  const std::string& curve,
                  int num_thresholds,
                  int slide_steps,
                  MetaTensor* auc,
                  MetaTensor* stat_pos_out,
                  MetaTensor* stat_neg_out,
                  MetaConfig config = MetaConfig());

void BatchNormInferMeta(const MetaTensor& x,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        const MetaTensor& mean,
                        const MetaTensor& variance,
                        float momentum,
                        float epsilon,
                        const std::string& data_layout,
                        bool is_test,
                        bool use_global_stats,
                        bool trainable_statistics,
                        bool fuse_with_relu,
                        MetaTensor* y,
                        MetaTensor* mean_out,
                        MetaTensor* variance_out,
                        MetaTensor* saved_mean,
                        MetaTensor* saved_variance,
                        MetaTensor* reserve_space,
                        MetaConfig config = MetaConfig());

void BatchNormInferInferMeta(const MetaTensor& x,
                             const MetaTensor& scale,
                             const MetaTensor& bias,
                             const MetaTensor& mean,
                             const MetaTensor& variance,
                             float momentum,
                             float epsilon,
                             const std::string& data_layout,
                             MetaTensor* y,
                             MetaTensor* mean_out,
                             MetaTensor* variance_out,
                             MetaConfig config = MetaConfig());

void BilinearTensorProductInferMeta(const MetaTensor& x,
                                    const MetaTensor& y,
                                    const MetaTensor& weight,
                                    paddle::optional<const MetaTensor&> bias,
                                    MetaTensor* out,
                                    MetaConfig config = MetaConfig());

void BroadcastTensorsInferMeta(const std::vector<const MetaTensor*>& x,
                               std::vector<MetaTensor*> out);

void ConcatInferMeta(const std::vector<const MetaTensor*>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());

void DeformableConvInferMeta(const MetaTensor& x,
                             const MetaTensor& offset,
                             const MetaTensor& filter,
                             paddle::optional<const MetaTensor&> mask,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& dilations,
                             int deformable_groups,
                             int groups,
                             int im2col_step,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());

void HierarchicalSigmoidInferMeta(const MetaTensor& x,
                                  const MetaTensor& w,
                                  const MetaTensor& label,
                                  paddle::optional<const MetaTensor&> path,
                                  paddle::optional<const MetaTensor&> code,
                                  paddle::optional<const MetaTensor&> bias,
                                  int num_classes,
                                  bool remote_prefetch,
                                  int trainer_id,
                                  const std::vector<int64_t>& height_sections,
                                  const std::vector<std::string>& epmap,
                                  const std::vector<std::string>& table_names,
                                  bool is_sparse,
                                  MetaTensor* out,
                                  MetaTensor* pre_out,
                                  MetaTensor* w_out);

void InterpolateInferMeta(
    const MetaTensor& x,
    paddle::optional<const MetaTensor&> out_size,
    paddle::optional<const std::vector<const MetaTensor*>> size_tensor,
    paddle::optional<const MetaTensor&> scale_tensor,
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

void LogspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       const MetaTensor& base,
                       MetaTensor* out);

void MeshgridInferMeta(const std::vector<const MetaTensor*>& inputs,
                       std::vector<MetaTensor*> outputs);

void MomentumInferMeta(const MetaTensor& param,
                       const MetaTensor& grad,
                       const MetaTensor& velocity,
                       const MetaTensor& learning_rate,
                       paddle::optional<const MetaTensor&> master_param,
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
                        paddle::optional<const MetaTensor&> rois_num,
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
                      paddle::optional<const MetaTensor&> mean_grad,
                      float epsilon,
                      float decay,
                      float momentum,
                      bool centered,
                      MetaTensor* param_out,
                      MetaTensor* moment_out,
                      MetaTensor* mean_square_out,
                      MetaTensor* mean_grad_out);

void RnnInferMeta(const MetaTensor& x,
                  const std::vector<const MetaTensor*>& pre_state,
                  const std::vector<const MetaTensor*>& weight_list,
                  paddle::optional<const MetaTensor&> sequence_length,
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

void SgdInferMeta(const MetaTensor& param,
                  const MetaTensor& learning_rate,
                  const MetaTensor& grad,
                  paddle::optional<const MetaTensor&> master_param,
                  bool multi_precision,
                  MetaTensor* param_out,
                  MetaTensor* master_param_out);

void StackInferMeta(const std::vector<const MetaTensor*>& x,
                    int axis,
                    MetaTensor* out);

void UnchangedMultiInferMeta(const std::vector<const MetaTensor*>& x,
                             std::vector<MetaTensor*> out);

void WarpctcInferMeta(const MetaTensor& logits,
                      const MetaTensor& label,
                      const paddle::optional<const MetaTensor&> logits_length,
                      const paddle::optional<const MetaTensor&> labels_length,
                      int blank,
                      bool norm_by_times,
                      MetaTensor* warpctc_grad,
                      MetaTensor* loss);

void WhereInferMeta(const MetaTensor& condition,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    MetaTensor* out);

void GraphReindexInferMeta(const MetaTensor& x,
                           const MetaTensor& neighbors,
                           const MetaTensor& count,
                           paddle::optional<const MetaTensor&> hashtable_value,
                           paddle::optional<const MetaTensor&> hashtable_index,
                           bool flag_buffer_hashtable,
                           MetaTensor* reindex_src,
                           MetaTensor* reindex_dst,
                           MetaTensor* out_nodes);

void GraphSampleNeighborsInferMeta(
    const MetaTensor& row,
    const MetaTensor& col_ptr,
    const MetaTensor& x,
    paddle::optional<const MetaTensor&> eids,
    paddle::optional<const MetaTensor&> perm_buffer,
    int sample_size,
    bool return_eids,
    bool flag_perm_buffer,
    MetaTensor* out,
    MetaTensor* out_count,
    MetaTensor* out_eids);

void Yolov3LossInferMeta(const MetaTensor& x,
                         const MetaTensor& gt_box,
                         const MetaTensor& gt_label,
                         const paddle::optional<const MetaTensor&> gt_score,
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

}  // namespace phi
