// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <functional>
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/operator_fusion/fusion_tracker/tracker.h"
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/pattern_fuser.h"

namespace cinn::fusion {

struct PatternExpr {
  std::vector<ir::Expr> exprs;
};
using PatternExprPtr = std::shared_ptr<PatternExpr>;

struct FusionInterpreter {
  FusionInterpreter(
      const FusionTrackerPtr& tracker,
      const std::unordered_map<pir::Operator*, ir::Expr>& lowered_expr)
      : tracker(tracker), lowered_expr(lowered_expr) {}

  std::unordered_map<pir::Operator*, ir::Expr> lowered_expr;
  std::unordered_map<std::string, PatternExprPtr> scope;
  FusionTrackerPtr tracker;

  PatternExpr Run();
};

}  // namespace cinn::fusion
