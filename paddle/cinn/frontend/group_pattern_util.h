#pragma once

#include "paddle/cinn/frontend/group_pattern.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"

namespace cinn::frontend {

GroupPattern GenerateGroupPatternFromFusionOp(const pir::FusionOp&);

}