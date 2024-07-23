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
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Add)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Add_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BitwiseAnd)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BitwiseAnd_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BitwiseXor)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BitwiseXor_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Complex)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Copysign)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Divide)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Divide_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(ElementwisePow)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Fmax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Fmin)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(GreaterEqual)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(GreaterEqual_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(GreaterThan)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(GreaterThan_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LessEqual)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LessEqual_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LessThan)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LessThan_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LogicalAnd)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LogicalAnd_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LogicalOr)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LogicalOr_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LogicalXor)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LogicalXor_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Maximum)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Minimum)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Multiply)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MultiplySr)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MultiplySr_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Multiply_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(NotEqual)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(NotEqual_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Remainder)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Remainder_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Subtract)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Subtract_)

}  // namespace paddle::dialect
