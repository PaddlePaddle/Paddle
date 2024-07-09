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

OP_DECLARE_INFER_SYMBOLIC_SHAPE(Accuracy)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Addmm)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Addmm_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(AddN)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Auc)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BicubicInterp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Bilinear)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BilinearInterp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Concat)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(CrossEntropyWithSoftmax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(CrossEntropyWithSoftmax_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FullWithTensor)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FlashAttn)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(GroupNorm)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Linspace)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LinearInterp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Logspace)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MemoryEfficientAttention)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Meshgrid)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(NearestInterp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(RoiAlign)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Stack)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TrilinearInterp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Where)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Where_)

}  // namespace paddle::dialect
