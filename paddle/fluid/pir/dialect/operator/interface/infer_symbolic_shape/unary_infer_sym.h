// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

OP_DECLARE_INFER_SYMBOLIC_SHAPE(Argmax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Argmin)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(AsComplex)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(AsReal)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cummax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cummin)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cumprod)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cumprod_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cumsum)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cumsum_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(DiagEmbed)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Diagonal)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Einsum)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Kthvalue)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Reshape)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Reshape_)

}  // namespace paddle::dialect
