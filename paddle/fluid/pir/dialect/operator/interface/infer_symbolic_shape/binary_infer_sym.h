// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace paddle::dialect {

OP_DECLARE_INFER_SYMBOLIC_SHAPE(Allclose)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Atan2)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BceLoss)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BceLoss_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Conv2d)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Conv3d)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Embedding)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LuUnpack)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(SparseWeightEmbedding)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(ExpandAs)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FillDiagonalTensor)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FillDiagonalTensor_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Gather)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(GatherNd)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Isclose)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(AccuracyCheck)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(IndexSample)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Kron)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MaskedSelect)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Matmul)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(ReduceAs)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Searchsorted)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TakeAlongAxis)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TopPSampling)

}  // namespace paddle::dialect

namespace cinn::dialect {
using paddle::dialect::IscloseOpInferSymbolicShape;
}
