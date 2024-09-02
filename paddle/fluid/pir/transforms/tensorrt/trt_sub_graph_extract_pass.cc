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

#include "paddle/fluid/pir/transforms/tensorrt/trt_sub_graph_extract_pass.h"
#include <glog/logging.h>
#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/fluid/pir/transforms/sub_graph_detector.h"

namespace {
using GroupOpsVec = std::vector<pir::Operation*>;

bool IsSupportedByTRT(const pir::Operation& op) {
  if (op.HasAttribute(paddle::dialect::kCanRunTrtAttr) &&
      op.attribute<pir::BoolAttribute>(paddle::dialect::kCanRunTrtAttr)
          .data()) {
    return true;
  }
  return false;
}

class TrtSubGraphExtractPass : public pir::Pass {
 public:
  explicit TrtSubGraphExtractPass(int min_group_size = 3)
      : pir::Pass("trt_sub_graph_extract_pass", 3),
        min_group_size_(min_group_size) {}

  void Run(pir::Operation* op) override {
    LOG(INFO) << "Running TrtSubGraphExtractPass with min_group_size: "
              << min_group_size_;
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        common::errors::InvalidArgument(
            "TrtSubGraphExtractPass should run on module op."));
    auto& block = module_op.block();

    std::vector<GroupOpsVec> groups =
        ::pir::SubgraphDetector(&block, IsSupportedByTRT)();
    AddStatistics(groups.size());
    for (auto& group_ops : groups) {
      if (group_ops.size() < static_cast<size_t>(min_group_size_)) {
        VLOG(4) << "Current group_ops.size(): " << group_ops.size()
                << ", will fallback to Paddle original graph";
        continue;
      }
      VLOG(4) << "Current group_ops.size(): " << group_ops.size()
              << ", will lower to TensorRT graph";
      ::pir::ReplaceWithGroupOp(&block, group_ops);
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  int min_group_size_;
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateTrtSubGraphExtractPass(int min_group_size) {
  return std::make_unique<TrtSubGraphExtractPass>(min_group_size);
}

}  // namespace pir

REGISTER_IR_PASS(trt_sub_graph_extract_pass, TrtSubGraphExtractPass);
