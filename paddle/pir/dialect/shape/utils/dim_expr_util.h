#pragma once

#include <optional>
#include "paddle/pir/core/builder.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"


namespace symbol {

::pir::Attribute ConvertDimExprToAttribute(::pir::Builder* builder, const DimExpr& dim_expr);
std::optional<DimExpr> ConvertAttributeToDimExpr(::pir::Attribute attribute);

}