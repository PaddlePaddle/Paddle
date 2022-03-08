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

void AddmmInferMeta(const MetaTensor& input,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    float alpha,
                    float beta,
                    MetaTensor* out);

void GatherNdGradInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& out_grad,
                           MetaTensor* x_grad);

void RangeInferMeta(const MetaTensor& Start,
                    const MetaTensor& End,
                    const MetaTensor& Step,
                    MetaTensor* out);

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

void LerpInferMeta(const MetaTensor& x,
                   const MetaTensor& y,
                   const MetaTensor& weight,
                   MetaTensor* out);

void LinspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       MetaTensor* out);

void GraphSendRecvInferMeta(const MetaTensor& x,
                            const MetaTensor& src_index,
                            const MetaTensor& dst_index,
                            const std::string& pool_type,
                            MetaTensor* out,
                            MetaTensor* dst_count);
}  // namespace phi
