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

#include <functional>

#include "paddle/pir/core/block.h"
#include "paddle/pir/dialect/shape/utils/shape_or_data_expr.h"

namespace pir {

class Block;
class IrContext;

}  // namespace pir

namespace cinn::dialect {

struct ShapeOrDataDimExprsAccessor {
  std::function<const symbol::ShapeOrDataDimExprs&(pir::Value)>
      GetShapeOrDataDimExprs;
  std::function<void(pir::Value, const symbol::ShapeOrDataDimExprs&)>
      SetShapeOrDataDimExprs;
};

// Returns true if at least one GenerateShapeOp rewrited.
bool MoveGenerateShapeOpsToPrologue(
    pir::IrContext* ir_context,
    pir::Block* block,
    const ShapeOrDataDimExprsAccessor& shape_or_data_dim_expr_accessor);

}  // namespace cinn::dialect
