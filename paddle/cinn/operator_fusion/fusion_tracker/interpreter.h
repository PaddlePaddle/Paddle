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
  explicit ScopeElement(const std::vector<FusibleOp> fusion_ops)
      : fusion_ops(fusion_ops) {}
  std::vector<FusibleOp> fusion_ops;
  void Extend(const std::vector<FusibleOp>& other) {
    fusion_ops.insert(fusion_ops.end(), other.begin(), other.end());
  }
};
using ScopeElementPtr = std::shared_ptr<ScopeElement>;
using cinn::hlir::framework::pir::trivial_fusion_detail::GetOutputTensor;

struct FusionInterpreter {
  FusionInterpreter(const FusionTrackerPtr& tracker,
                    const std::vector<::pir::Operation*>& ops,
                    const std::vector<FusibleOp>& init_fusible_op)
      : tracker(tracker), initialized_lowered_op(init_fusible_op) {
    for (size_t i = 0; i < ops.size(); i++) {
      if (ops[i]->name() == "cinn_op.yield_store" ||
          ops[i]->name() == "pd_op.assign_out_") {
        global_var_names.push_back(GetOutputTensor(init_fusible_op[i])->name);
      }
    }
    VLOG(4) << "Create FusionInterpreter, Tracker is:\n" << tracker->DebugStr();
    VLOG(4) << "Global var names: " << utils::Join(global_var_names, ", ");
  }

  std::vector<FusibleOp> initialized_lowered_op;
  std::vector<std::string> global_var_names;
  std::unordered_map<std::string, ScopeElementPtr> scope;
  FusionTrackerPtr tracker;

  std::vector<ir::Expr> ret_expr;
  std::vector<ir::Expr> Run();
};
}  // namespace cinn::fusion
