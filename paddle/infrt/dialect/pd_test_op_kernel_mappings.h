#pragma once
#include <mlir/IR/PatternMatch.h>
#include "paddle/infrt/dialect/infrt_base.h"
#include "paddle/infrt/dialect/pd_ops.h"

namespace infrt {
namespace pd {

#include "paddle/infrt/dialect/pd_test_op_kernel_mappings.hpp.inc"

}  // namespace pd
}  // namespace infrt

