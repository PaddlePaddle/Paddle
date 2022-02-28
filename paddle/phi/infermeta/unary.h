/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/meta_tensor.h"

namespace phi {

class MetaConfig;

// Common InferMeta Functions for unary operators, The format like:
//
//   void [FunctionDesc|OpName]InferMeta(const MetaTensor& x, ..., MetaTensor*
//   out) {}
//
// NOTE: The name "InferShape" may be not appropriate. "InferMeta" may be good.
// Because functions in this file not only can infer shape, but also need
// infer lod or other useful data.

void UnchangedInferMeta(const MetaTensor& x, MetaTensor* out);

// meta x -> out without change, check if axis in range [-Rank(x), Rank(x)-1]
void UnchangedInferMetaCheckAxis(const MetaTensor& x,
                                 int axis,
                                 MetaTensor* out);

void FlattenInferMeta(const MetaTensor& x,
                      int start_axis,
                      int stop_axis,
                      MetaTensor* out);

void GumbelSoftmaxInferMeta(const MetaTensor& x,
                            float temperature,
                            bool hard,
                            int axis,
                            MetaTensor* out);

void CastInferMeta(const MetaTensor& x, DataType out_dtype, MetaTensor* out);

void CholeskyInferMeta(const MetaTensor& x, bool upper, MetaTensor* out);

void CopyToInferMeta(const MetaTensor& x,
                     Backend backend,
                     bool blocking,
                     MetaTensor* out);

void CreateLikeInferMeta(const MetaTensor& x, DataType dtype, MetaTensor* out);

void IncrementInferMeta(const MetaTensor& x, float value, MetaTensor* out);

void InferMetaFromVecValue(const MetaTensor& x,
                           const std::vector<int64_t>& shape,
                           MetaTensor* out);

void MultinomialInferMeta(const MetaTensor& x,
                          int num_samples,
                          bool replacement,
                          MetaTensor* out);

void ReshapeInferMeta(const MetaTensor& x,
                      const ScalarArray& shape,
                      MetaTensor* out,
                      MetaConfig config = MetaConfig());

void ReshapeWithXShapeInferMeta(const MetaTensor& x,
                                const ScalarArray& shape,
                                MetaTensor* xshape,
                                MetaTensor* out,
                                MetaConfig config = MetaConfig());

void ReduceInferMetaBase(const MetaTensor& x,
                         const std::vector<int64_t>& axis,
                         bool keep_dim,
                         DataType dtype,
                         MetaTensor* out);

void ReduceInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keep_dim,
                     MetaTensor* out);

void SumInferMeta(const MetaTensor& x,
                  const std::vector<int64_t>& axis,
                  DataType dtype,
                  bool keep_dim,
                  MetaTensor* out);

void TransferLayoutInferMeta(const MetaTensor& x,
                             DataLayout layout,
                             MetaTensor* out);

void SplitInferMeta(const MetaTensor& x_meta,
                    const ScalarArray& num_or_sections,
                    const Scalar& axis,
                    std::vector<MetaTensor>* out,
                    MetaConfig config = MetaConfig());

void UnbindInferMeta(const MetaTensor& x,
                     int axis,
                     std::vector<MetaTensor>* outs);
void TraceInferMeta(
    const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out);

void UnfoldInferMeta(const MetaTensor& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());

void DiagInferMeta(const MetaTensor& x,
                   int offset,
                   float padding_value,
                   MetaTensor* out);

}  // namespace phi
