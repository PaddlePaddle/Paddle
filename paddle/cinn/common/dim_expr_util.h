#pragma once

#include "paddle/pir/dialect/shape/utils/dim_expr.h"

namespace cinn::common {

symbol::DimExpr SimplifyDimExpr(const symbol::DimExpr&);

}