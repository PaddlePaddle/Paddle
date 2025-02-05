// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include <cstdint>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class DeadCodeEliminationPass : public pir::Pass {
 public:
  DeadCodeEliminationPass() : pir::Pass("dead_code_elimination_pass", 0) {}

  void Run(pir::Operation* op) override {
    VLOG(6) << "apply dead_code_elimination_pass";
    int64_t num_erasers{0};
    std::vector<std::string> deleted_vars;
    bool updated{true};
    while (updated) {
      int64_t pre_num_erasers = num_erasers;
      EraseOp(*op->GetParentProgram()->block(), &num_erasers, &deleted_vars);
      updated = pre_num_erasers != num_erasers;
    }
    if (Has(pir::Pass::kParamScopeAttr)) {
      auto scope = &Get<paddle::framework::Scope>(pir::Pass::kParamScopeAttr);
      if (deleted_vars.size() > 0) {
        scope->EraseVars(deleted_vars);
      }
    }
    AddStatistics(num_erasers);
  }

 private:
  void EraseOp(const pir::Block& block,
               int64_t* num_erasers,
               std::vector<std::string>* deleted_vars) {
    std::vector<pir::Operation*> deleted_ops;
    for (auto& op : block) {
      if (op.HasTrait<pir::SideEffectTrait>() ||
          op.isa<paddle::dialect::DataOp>() ||
          op.isa<paddle::dialect::WhileOp>() ||
          paddle::dialect::IsCustomOp(&op) ||
          paddle::dialect::IsInplaceOp(&op)) {
        continue;
      }
      if (op.use_empty()) {
        deleted_ops.push_back(&op);
      }
    }

    for (auto* op : deleted_ops) {
      if (op->isa<pir::ParameterOp>()) {
        auto parameter_op = op->dyn_cast<pir::ParameterOp>();
        deleted_vars->push_back(parameter_op.param_name());
      } else if (op->isa<pir::ConstantTensorOp>()) {
        auto constant_tensor_op = op->dyn_cast<pir::ConstantTensorOp>();
        deleted_vars->push_back(constant_tensor_op.tensor_name());
      }
      op->Erase();
      VLOG(4) << "erase op: " << op->name();
      (*num_erasers)++;
    }

    if (deleted_ops.empty()) {
      for (auto& op : block) {
        for (size_t i = 0; i < op.num_regions(); ++i) {
          auto& inner_region = op.region(i);
          for (auto& inner_block : inner_region) {
            EraseOp(inner_block, num_erasers, deleted_vars);
          }
        }
      }
    } else {
      EraseOp(block, num_erasers, deleted_vars);
    }
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateDeadCodeEliminationPass() {
  return std::make_unique<DeadCodeEliminationPass>();
}

}  // namespace pir

REGISTER_IR_PASS(dead_code_elimination_pass, DeadCodeEliminationPass);
