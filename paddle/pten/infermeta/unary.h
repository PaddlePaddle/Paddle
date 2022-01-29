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
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/infermeta_utils.h"
#include "paddle/pten/core/meta_tensor.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

class MetaConfig;

// Common InferMeta Functions for unary operators, The format like:
//
//   void [OpName]InferMeta(const MetaTensor& x, ..., MetaTensor* out) {}
//
// NOTE: The name "InferShape" may be not appropriate. "InferMeta" may be good.
// Because functions in this file not only can infer shape, but also need
// infer lod or other useful data.

// TODO(chenweihang): update all InferMeta function format in next pr,
// now add UnchangedInferMetaNew for test new format
void UnchangedInferMetaNew(MetaConfig config,
                           const MetaTensor& x,
                           MetaTensor* out);

DenseTensorMeta UnchangedInferMeta(const DenseTensorMeta& x_meta);

DenseTensorMeta ReductionInferMeta(const DenseTensorMeta& x_meta);

DenseTensorMeta FlattenInferMeta(const DenseTensorMeta& x_meta,
                                 int start_axis,
                                 int stop_axis);
DenseTensorMeta CastInferMeta(const DenseTensorMeta& x_meta,
                              const DataType out_dtype);

DenseTensorMeta CreateLikeInferMeta(const DenseTensorMeta& x_meta,
                                    DataType dtype,
                                    DataLayout layout);

DenseTensorMeta InferMetaFromVecValue(const DenseTensorMeta& x_meta,
                                      const std::vector<int64_t>& shape);

DenseTensorMeta ReshapeInferMeta(const DenseTensorMeta& x_meta,
                                 const ScalarArray& shape);

DenseTensorMeta ReduceInferMeta(const DenseTensorMeta& x_meta,
                                const std::vector<int64_t>& axis,
                                bool keep_dim,
                                DataType dtype = DataType::UNDEFINED);

DenseTensorMeta SumInferMeta(const DenseTensorMeta& x_meta,
                             const std::vector<int64_t>& axis,
                             DataType dtype,
                             bool keep_dim);

DenseTensorMeta TransferLayoutInferMeta(const DenseTensorMeta& x_meta,
                                        DataLayout layout);

}  // namespace pten
