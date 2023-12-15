#pragma once

#include "paddle/pir/dialect/shape/utils/dim_expr.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn::common {

struct DimExprConverter final {

  ir::Expr ConvertToIrExpr(const symbol::DimExpr&) const;

};

}