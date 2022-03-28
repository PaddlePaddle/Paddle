// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/infrt/dialect/pd/ir/pd_ops.h"

#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

#include "paddle/infrt/dialect/core/ir/core_dialect.h"
#include "paddle/infrt/dialect/pd/ir/pd_opsDialect.cpp.inc"
#define GET_OP_CLASSES
#include "paddle/infrt/dialect/pd/ir/pd_ops.cpp.inc"  // NOLINT
#define GET_OP_CLASSES
#include "paddle/infrt/dialect/pd/ir/pd_extra_ops.cpp.inc"  // NOLINT

namespace infrt {
namespace pd {
void PaddleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/pd/ir/pd_ops.cpp.inc"  // NOLINT
      ,
#define GET_OP_LIST
#include "paddle/infrt/dialect/pd/ir/pd_extra_ops.cpp.inc"  // NOLINT
      >();
}
}  // namespace pd
}  // namespace infrt
