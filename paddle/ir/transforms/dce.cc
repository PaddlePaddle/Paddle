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

#include "paddle/ir/transforms/dce.h"
#include <memory>
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/pass/pass.h"

namespace {

// TODO(wilber): After support SideEffectTrait, Only NoSideEffectTrait op can be
// removed by dce pass.
// Now just a naive implementation.
class DCEPass : public ir::Pass {
 public:
  DCEPass() : ir::Pass("DCEPass", 0) {}

  void Run(ir::Operation *op) override {
    auto module_op = op->dyn_cast<ir::ModuleOp>();
    IR_ENFORCE(module_op, "DCEPass should run on module op.");
    auto *block = module_op.block();
    std::vector<ir::Operation> erased_op;
    for (auto it = block->begin(); it != block->end(); ++it) {
      // TODO(wilber): Support NoSideEffect trait.
      // if (!(*it)->HasTrait<NoSideEffect>()) continue;

      bool use_empty = true;
      for (uint32_t i = 0; i < (*it)->num_results(); ++i) {
        use_empty &= (*it)->result(i).use_empty();
      }

      // TODO(wilber): Support end trait.
      if (use_empty && (*it)->name() != "pd.fetch") {
        erased_op.push_back(**it);
      }
    }

    for (auto ep : erased_op) block->erase(ep);
  }

  bool CanApplyOn(ir::Operation *op) const override {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }
};

}  // namespace

namespace ir {

std::unique_ptr<Pass> CreateDCEPass() { return std::make_unique<DCEPass>(); }

}  // namespace ir
