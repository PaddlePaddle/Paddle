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

#include "paddle/pir/transforms/dead_code_elimination_pass.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {

// TODO(wilber): After support SideEffectTrait, Only NoSideEffectTrait op can be
// removed by dce pass.
// Now just a naive implementation.
class DeadCodeEliminationPass : public pir::Pass {
 public:
  DeadCodeEliminationPass() : pir::Pass("dead_code_elimination", 0) {}

  void Run(pir::Operation *op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "DcePass should run on module op.");
    auto *block = module_op.block();
    std::vector<pir::Operation *> erased_op;
    for (auto &op : *block) {
      // TODO(wilber): Support NoSideEffect trait.
      // if (!op->HasTrait<NoSideEffect>()) continue;

      bool use_empty = true;
      for (uint32_t i = 0; i < op->num_results(); ++i) {
        use_empty &= op->result(i).use_empty();
      }
      // TODO(wilber): Support Terminator trait.
      if (use_empty && op->name() != "pd_op.fetch") {
        erased_op.push_back(op);
      }
    }

    for (auto *op : erased_op) {
      if (op->dyn_cast<pir::GetParameterOp>()) {
        // Delete parameter from program.
        pir::GetParameterOp get_parameter_op =
            op->dyn_cast<pir::GetParameterOp>();
        get_parameter_op->GetParentProgram()->parameters().erase(
            get_parameter_op->attributes()
                .at(get_parameter_op.attributes_name[0])
                .dyn_cast<pir::StrAttribute>()
                .AsString());
      }
      block->erase(*op);
    }
  }

  bool CanApplyOn(pir::Operation *op) const override {
    return op->isa<::pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateDeadCodeEliminationPass() {
  return std::make_unique<DeadCodeEliminationPass>();
}

}  // namespace pir

REGISTER_IR_PASS(dead_code_elimination, DeadCodeEliminationPass);
