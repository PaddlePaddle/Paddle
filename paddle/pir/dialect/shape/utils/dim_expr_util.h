#pragma once

#include <optional>
#include "paddle/pir/core/builder.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"


namespace symbol {

::pir::Attribute ConvertDimExprToAttribute(::pir::Builder* builder, const DimExpr& dim_expr);
std::optional<DimExpr> ConvertAttributeToDimExpr(::pir::Attribute attribute);

std::optional<DimExpr> SubstituteDimExpr(
    const DimExpr& dim_expr,
    const std::function<std::optional<DimExpr>(const std::string& symbol_name)>& DimExpr4SymbolName);

std::function<std::optional<DimExpr>(const std::string& symbol_name)>
MakeGetterDimExpr4SymbolName(
    const std::vector<std::tuple<std::string/*symbol_name*/, int/*in_tensor_idx*/, int/*in_tensor_dim_idx*/>>& symbol_bindings,
    const std::function<std::optional<DimExpr>(int in_tensor_idx, int in_tensor_dim_idx)>& DimExpr4InputDim);

}