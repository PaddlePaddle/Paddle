// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#pragma once
#include <optional>
#include <unordered_map>
#include <vector>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
namespace cinn {
namespace common {

// `MergeMulMod` is used to simplify the expression for Max Level of
// `IndexExpr::OptLevel` When it simplifies addition, because of the commutative
// law, it collects all the operands of addition and then tries to simplify each
// element. It has a time complexity of O3 and cannot be enabled by default for
// performance reasons.
ir::IndexExpr MergeMulMod(const ir::IndexExpr& expr);

std::optional<ir::IndexExpr> SimplifyCornerCase(const ir::IndexExpr& expr);
std::optional<ir::IndexExpr> SimplifyAddCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs);
std::optional<ir::IndexExpr> SimplifyMulCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs);
std::optional<ir::IndexExpr> SimplifyDivCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs);
std::optional<ir::IndexExpr> SimplifyModCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs);
}  // namespace common
}  // namespace cinn
