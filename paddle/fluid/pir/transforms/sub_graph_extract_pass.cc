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

#include "paddle/fluid/pir/transforms/sub_graph_extract_pass.h"

#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/cinn/hlir/framework/pir/utils.h"

#include "paddle/fluid/pir/transforms/sub_graph_detector.h"

namespace {
using GroupOpsVec = std::vector<pir::Operation*>;

bool IsMatmulOp(const pir::Operation& op) {
  return op.name() == "pd_op.matmul";
}

class SubGraphExtractPass : public pir::Pass {
 public:
  SubGraphExtractPass()
      : pir::Pass("sub_graph_extract_pass", /*opt_level=*/1) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        common::errors::InvalidArgument(
            "sub_graph_extract_pass should run on module op."));
    auto& block = module_op.block();

    std::vector<GroupOpsVec> groups =
        ::pir::DetectSubGraphs(&block, IsMatmulOp);
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

std::unique_ptr<Pass> CreateSubGraphExtractPass() {
  return std::make_unique<SubGraphExtractPass>();
}

}  // namespace pir

REGISTER_IR_PASS(sub_graph_extract_pass, SubGraphExtractPass);
