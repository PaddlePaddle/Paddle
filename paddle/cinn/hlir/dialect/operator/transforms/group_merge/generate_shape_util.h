#pragma once

#include "paddle/pir/core/block.h"

#include "paddle/pir/dialect/shape/utils/dim_expr.h"

#include <functional>

namespace pir {

class Block;
class IrContext;

}

namespace cinn::dialect {

struct ShapeOrDataDimExprsCtx {
  std::function<const symbol::ShapeOrDataDimExprs&(pir::Value)> GetShapeOrDataDimExprs;
  std::function<void(pir::Value, const symbol::ShapeOrDataDimExprs&)> SetShapeOrDataDimExprs;
};

void RewriteGenerateShapeOpToRunFirst(
    pir::IrContext* ir_context,
    pir::Block* block,
    const ShapeOrDataDimExprsCtx& shape_or_data_dim_expr_ctx);

}