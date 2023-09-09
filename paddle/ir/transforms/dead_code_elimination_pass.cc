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

#include "paddle/ir/transforms/dead_code_elimination_pass.h"

#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_registry.h"

namespace {

// TODO(wilber): After support SideEffectTrait, Only NoSideEffectTrait op can be
// removed by dce pass.
// Now just a naive implementation.
class DeadCodeEliminationPass : public ir::Pass {
 public:
  DeadCodeEliminationPass() : ir::Pass("dead_code_elimination", 0) {}

  void Run(ir::Operation *op) override {
    auto module_op = op->dyn_cast<ir::ModuleOp>();
    IR_ENFORCE(module_op, "DcePass should run on module op.");
    auto *block = module_op.block();
    std::vector<ir::Operation *> erased_op;
    for (auto &op : *block) {
      // TODO(wilber): Support NoSideEffect trait.
      // if (!op->HasTrait<NoSideEffect>()) continue;

      bool use_empty = true;
      for (uint32_t i = 0; i < op->num_results(); ++i) {
        use_empty &= op->result(i).use_empty();
      }
      // TODO(wilber): Support Terminator trait.
      if (use_empty && op->name() != "pd.fetch") {
        erased_op.push_back(op);
      }
    }

    for (auto *op : erased_op) {
      if (op->dyn_cast<ir::GetParameterOp>()) {
        // Delete parameter from program.
        ir::GetParameterOp get_parameter_op =
            op->dyn_cast<ir::GetParameterOp>();
        get_parameter_op->GetParentProgram()->parameters().erase(
            get_parameter_op->attributes()
                .at(get_parameter_op.attributes_name[0])
                .dyn_cast<ir::StrAttribute>()
                .AsString());
      }
      block->erase(*op);
    }
  }

  bool CanApplyOn(ir::Operation *op) const override {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }
};

}  // namespace

namespace ir {

std::unique_ptr<Pass> CreateDeadCodeEliminationPass() {
  return std::make_unique<DeadCodeEliminationPass>();
}

}  // namespace ir

REGISTER_IR_PASS(dead_code_elimination, DeadCodeEliminationPass);
