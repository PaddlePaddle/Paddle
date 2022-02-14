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
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/meta_tensor.h"

namespace pten {

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

void FlattenInferMeta(const MetaTensor& x,
                      int start_axis,
                      int stop_axis,
                      MetaTensor* out);

void CastInferMeta(const MetaTensor& x, DataType out_dtype, MetaTensor* out);

void CreateLikeInferMeta(const MetaTensor& x,
                         DataType dtype,
                         DataLayout layout,
                         MetaTensor* out);

void InferMetaFromVecValue(const MetaTensor& x,
                           const std::vector<int64_t>& shape,
                           MetaTensor* out);

void ReshapeInferMeta(const MetaTensor& x,
                      const ScalarArray& shape,
                      MetaTensor* out);

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
}  // namespace pten
