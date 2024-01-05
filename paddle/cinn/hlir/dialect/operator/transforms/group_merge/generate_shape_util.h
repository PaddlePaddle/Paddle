#pragma once

#include "paddle/pir/core/block.h"

#include "paddle/pir/dialect/shape/utils/dim_expr.h"

#include <functional>

namespace pir {

class Block;
class IrContext;

}

namespace cinn::dialect {

struct ShapeOrDataDimExprsAccessor {
  std::function<const symbol::ShapeOrDataDimExprs&(pir::Value)> GetShapeOrDataDimExprs;
  std::function<void(pir::Value, const symbol::ShapeOrDataDimExprs&)> SetShapeOrDataDimExprs;
};

// Returns true if at least one GenerateShapeOp rewrited.
bool RewriteGenerateShapeOpToRunFirst(
    pir::IrContext* ir_context,
    pir::Block* block,
    const ShapeOrDataDimExprsAccessor& shape_or_data_dim_expr_accessor);

}