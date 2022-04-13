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

#include "paddle/phi/core/meta_tensor.h"

namespace phi {

// Common InferMeta Functions for ternary operators, The format like:
//
//   1. void [FunctionDesc|OpName]InferMeta(const MetaTensor& x,
//                                          const MetaTensor& y,
//                                          const MetaTensor& z,
//                                          ...,
//                                          MetaTensor* out) {}
//
// NOTE: The name "InferShape" may be not appropriate. "InferMeta" may be good.
//   Because functions in this file not only can infer shape, but also need
//   infer lod or other useful data.
//
// The InferMeta Functions in this file are arranged in alphabetic order.

void AccuracyInferMeta(const MetaTensor& out,
                       const MetaTensor& indice,
                       const MetaTensor& label,
                       MetaTensor* accuracy,
                       MetaTensor* correct,
                       MetaTensor* total,
                       MetaConfig config = MetaConfig());

void AddmmInferMeta(const MetaTensor& input,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    float alpha,
                    float beta,
                    MetaTensor* out);

void ArangeInferMeta(const MetaTensor& start,
                     const MetaTensor& end,
                     const MetaTensor& step,
                     MetaTensor* out);

void GraphSendRecvInferMeta(const MetaTensor& x,
                            const MetaTensor& src_index,
                            const MetaTensor& dst_index,
                            const std::string& pool_type,
                            int64_t out_size,
                            MetaTensor* out,
                            MetaTensor* dst_count);

void LayerNormInferMeta(const MetaTensor& x,
                        paddle::optional<const MetaTensor&> scale,
                        paddle::optional<const MetaTensor&> bias,
                        float epsilon,
                        int begin_norm_axis,
                        bool is_test,
                        MetaTensor* out,
                        MetaTensor* mean,
                        MetaTensor* variance,
                        MetaConfig config = MetaConfig());

void LayerNormGradInferMeta(const MetaTensor& x,
                            paddle::optional<const MetaTensor&> y,
                            paddle::optional<const MetaTensor&> z,
                            MetaTensor* dx,
                            MetaTensor* dy,
                            MetaTensor* dz);

void LerpInferMeta(const MetaTensor& x,
                   const MetaTensor& y,
                   const MetaTensor& weight,
                   MetaTensor* out);

void LinspaceRawInferMeta(const MetaTensor& start,
                          const MetaTensor& stop,
                          const MetaTensor& number,
                          MetaTensor* out);

void LinspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       DataType dtype,
                       MetaTensor* out);

void NllLossRawInferMeta(const MetaTensor& input,
                         const MetaTensor& label,
                         paddle::optional<const MetaTensor&> weight,
                         int64_t ignore_index,
                         const std::string& reduction,
                         MetaTensor* out,
                         MetaTensor* total_weight,
                         MetaConfig config = MetaConfig());

void PutAlongAxisInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& value,
                           int axis,
                           const std::string& reduce,
                           MetaTensor* out);

void RoiAlignInferMeta(const MetaTensor& x,
                       const MetaTensor& boxes,
                       paddle::optional<const MetaTensor&> boxes_num,
                       int pooled_height,
                       int pooled_width,
                       float spatial_scale,
                       int sampling_ratio,
                       bool aligned,
                       MetaTensor* out,
                       MetaConfig config = MetaConfig());

void RoiPoolInferMeta(const MetaTensor& x,
                      const MetaTensor& boxes,
                      paddle::optional<const MetaTensor&> boxes_num,
                      int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      MetaTensor* out,
                      MetaTensor* arg_max);

void ScatterInferMeta(const MetaTensor& x,
                      const MetaTensor& index,
                      const MetaTensor& updates,
                      bool overwrite,
                      MetaTensor* out);

void ScatterNdAddInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& updates,
                           MetaTensor* out);

void ViterbiDecodeInferMeta(const MetaTensor& input,
                            const MetaTensor& transition,
                            const MetaTensor& length,
                            bool include_bos_eos_tag,
                            MetaTensor* scores,
                            MetaTensor* path,
                            MetaConfig config = MetaConfig());

}  // namespace phi
