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

namespace paddle {
namespace dialect {

const symbol::ShapeOrDataDimExprs& GetInputShape(
    const pir::InferSymbolicShapeContext* infer_context,
    pir::Operation* op,
    int index);

const symbol::ShapeOrDataDimExprs& GetOutputShape(
    const pir::InferSymbolicShapeContext* infer_context,
    pir::Operation* op,
    int index);

symbol::ShapeOrDataDimExprs GetGradVarShapeFromOutput(
    const pir::InferSymbolicShapeContext* infer_context,
    pir::Operation* op,
    int index);

symbol::ShapeOrDataDimExprs GetGradVarShapeFromInput(
    const pir::InferSymbolicShapeContext* infer_context,
    pir::Operation* op,
    int index);

}  // namespace dialect
}  // namespace paddle
