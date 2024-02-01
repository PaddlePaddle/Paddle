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

#include "paddle/fluid/pir/transforms/build_cinn_pass.h"

#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/fluid/pir/transforms/sub_graph_detector.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {
using GroupOpsVec = std::vector<pir::Operation*>;
using CompatibleInfo = cinn::hlir::framework::pir::CompatibleInfo;

class BuildCinnPass : public pir::Pass {
 public:
  BuildCinnPass() : pir::Pass("build_cinn_pass", /*opt_level=*/1) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "build_cinn_pass should run on module op.");
    auto& block = module_op.block();

    std::vector<GroupOpsVec> groups =
        ::pir::SubgraphDetector(&block, CompatibleInfo::IsSupportCinn)();
    AddStatistics(groups.size());
    for (auto& group_ops : groups) {
      VLOG(4) << "current group_ops.size(): " << group_ops.size();
      ::pir::ReplaceWithGroupOp(&block, group_ops);
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateBuildCinnPass() {
  return std::make_unique<BuildCinnPass>();
}

}  // namespace pir

REGISTER_IR_PASS(build_cinn_pass, BuildCinnPass);
