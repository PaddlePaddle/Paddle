// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/cuda_graph_extract_pass.h"

#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/fluid/pir/transforms/sub_graph_detector.h"

COMMON_DECLARE_string(cuda_graph_blacklist);

namespace pir {
using GroupOpsVec = std::vector<Operation*>;

class CudaGraphExtractPass : public Pass {
 public:
  CudaGraphExtractPass() : Pass("cuda_graph_extract_pass", /*opt_level=*/1) {}

  void Run(Operation* op) override {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto module_op = op->dyn_cast<ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        common::errors::InvalidArgument(
            "sub_graph_extract_pass should run on module op."));
    auto& block = module_op.block();

    auto IsSupportCudaGraph = [](const Operation& op) {
      static const std::unordered_set<std::string> UNSUPPORTED_OPS = {
          "pd_op.data", "builtin.shadow_output"};
      static const std::unordered_set<std::string> CUDA_GRAPH_BLACKLIST = [] {
        std::regex re(",");
        std::sregex_token_iterator it(FLAGS_cuda_graph_blacklist.begin(),
                                      FLAGS_cuda_graph_blacklist.end(),
                                      re,
                                      -1);
        std::sregex_token_iterator end;
        return std::unordered_set<std::string>(it, end);
      }();
      return UNSUPPORTED_OPS.count(op.name()) == 0 &&
             CUDA_GRAPH_BLACKLIST.count(op.name()) == 0;
    };

    std::vector<GroupOpsVec> groups =
        DetectSubGraphs(&block, IsSupportCudaGraph);

    for (auto& group_ops : groups) {
      VLOG(4) << "current cuda_group count : " << group_ops.size();
      ReplaceWithCudaGraphOp(&block, group_ops);
    }
#endif
  }

  bool CanApplyOn(Operation* op) const override {
    return op->isa<ModuleOp>() && op->num_regions() > 0;
  }

 private:
  void ReplaceWithCudaGraphOp(Block* block, const GroupOpsVec& group_ops) {
    IrContext* ctx = IrContext::Instance();
    Builder builder = Builder(ctx, block);
    const std::vector<Value> outputs = AnalysisOutputs(group_ops, false);

    // step 1: Analysis and insert group op before insert_point.
    auto* insert_point = FindInsertPoint(group_ops, outputs);
    MoveUpstreamOpBeforeGroup(group_ops, block, insert_point);
    builder.set_insertion_point(insert_point);
    VLOG(6) << "Insert GroupOp after " << insert_point->name();

    // step 2: Replace the old op with CudaGraphOp.
    auto cuda_graph_op = [&]() -> paddle::dialect::CudaGraphOp {
      std::vector<Type> output_types;
      for (auto& value : outputs) output_types.emplace_back(value.type());

      auto group_op = builder.Build<paddle::dialect::CudaGraphOp>(output_types);
      for (auto op : group_ops) {
        op->MoveTo(group_op.block(), group_op.block()->end());
      }
      return group_op;
    }();

    // step 3: Replace outputs of inner ops
    const std::vector<Value> group_outs = cuda_graph_op->results();
    std::unordered_set<Operation*> inner_ops(group_ops.begin(),
                                             group_ops.end());
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].ReplaceUsesWithIf(group_outs[i], [&inner_ops](OpOperand op) {
        return !inner_ops.count(op.owner());
      });
    }

    // step 4: Insert YieldOp for outputs
    builder.SetInsertionPointToBlockEnd(cuda_graph_op.block());
    builder.Build<YieldOp>(outputs);
  }
};

std::unique_ptr<Pass> CreateCudaGraphExtractPass() {
  return std::make_unique<CudaGraphExtractPass>();
}

}  // namespace pir

REGISTER_IR_PASS(cuda_graph_extract_pass, pir::CudaGraphExtractPass);
