#pragma once

#include <optional>
#include "paddle/pir/core/builder.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/common/dim_expr_simplify.h"

namespace cinn::common {

symbol::DimExpr SubstituteDimExpr(
    const symbol::DimExpr& dim_expr,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>& pattern_to_replacement);

}