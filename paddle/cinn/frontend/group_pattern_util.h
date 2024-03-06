#pragma once

#include "paddle/cinn/frontend/group_pattern.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"

namespace cinn::frontend {

struct ErrorGroupPattern {
  const pir::Operation* op;
  std::string error_string;
};

std::variant<GroupPattern, ErrorGroupPattern> GenerateGroupPatternFromFusionOp(const cinn::dialect::FusionOp&);

}