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

std::vector<DDim> GetMetaTensorsDim(const std::vector<MetaTensor*>& tensors);

void AdadeltaInferMeta(const MetaTensor& param,
                       const MetaTensor& grad,
                       const MetaTensor& avg_squared_grad,
                       const MetaTensor& avg_squared_update,
                       float rho,
                       float epsilon,
                       MetaTensor* param_out,
                       MetaTensor* avg_squared_grad_out,
                       MetaTensor* avg_squared_update_out);

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

void BroadcastTensorsInferMeta(const std::vector<MetaTensor*>& x,
                               std::vector<MetaTensor*> out);

void ConcatInferMeta(const std::vector<MetaTensor*>& x,
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

void MultiDotInferMeta(const std::vector<MetaTensor*>& x, MetaTensor* out);

void PsroiPoolInferMeta(const MetaTensor& x,
                        const MetaTensor& rois,
                        paddle::optional<const MetaTensor&> rois_num,
                        int pooled_height,
                        int pooled_width,
                        int output_channels,
                        float spatial_scale,
                        MetaTensor* out);

void WhereInferMeta(const MetaTensor& condition,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    MetaTensor* out);

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
