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
#include "paddle/cinn/operator_fusion/fusion_tracker/expr_utils.h"
#include "paddle/cinn/operator_fusion/fusion_tracker/tracker.h"
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/pattern_fuser.h"

namespace cinn::fusion {

struct ScopeElement {
  ScopeElement() = default;
  ScopeElemnt(const std::vector<FusionOp> fusion_ops)
      : fusion_ops(fusion_ops) {}
  std::vector<FusionOp> fusion_ops;
};
using ScopeElementPtr = std::shared_ptr<ScopeElement>;

ScopeElementPtr CombineScopeElement(const ScopeElementPtr& a,
                                    const ScopeElementPtr& b) {
  return make_shared<ScopeElemnt>(ConcatVector(a->fusion_ops, b->fusion_ops));
}

struct FusionInterpreter {
  FusionInterpreter(
      const FusionTrackerPtr& tracker,
      const std::unordered_map<pir::Operation*, FusionOp>& lowered_expr)
      : tracker(tracker), lowered_expr(lowered_expr) {}

  std::unordered_map<pir::Operation*, FusionOp> lowered_expr;
  std::unordered_map<std::string, ScopeElementPtr> scope;
  FusionTrackerPtr tracker;

  std::vector<ir::Expr> ret_expr;
  std::vector<ir::Expr> Run();
};

template <typename T>
std::shared_ptr<T> dynamic_cast_instr_with_err(FusionInstrPtr instr) {
  auto chile_instr = std::dynamic_pointer_cast<T>(instr);
  if (!chile_instr) PADDLE_THROW("Cast Fusion Instr Failed.");
  return chile_instr;
}

}  // namespace cinn::fusion
