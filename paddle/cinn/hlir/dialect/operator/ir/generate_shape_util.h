#pragma once

#include <optional>
#include "paddle/pir/core/builder.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"


namespace cinn::dialect {

::pir::Attribute ConvertDimExprToAttribute(pir::IrContext* ctx, const symbol::DimExpr& dim_expr);
std::optional<symbol::DimExpr> ConvertAttributeToDimExpr(::pir::Attribute attribute);

std::optional<symbol::DimExpr> SubstituteDimExpr(
    const symbol::DimExpr& dim_expr,
    const std::function<std::optional<symbol::DimExpr>(const std::string& symbol_name)>& DimExpr4SymbolName);

std::function<std::optional<symbol::DimExpr>(const std::string& symbol_name)>
MakeGetterDimExpr4SymbolName(
    const GenerateShapeOp::SymbolBindings& symbol_bindings,
    const std::function<const symbol::ShapeOrDataDimExprs&(int in_tensor_idx)>& DimExpr4InputDim);

}