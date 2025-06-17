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

#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/fluid/pir/transforms/sub_graph_detector.h"

namespace {
using GroupOpsVec = std::vector<pir::Operation*>;

class CudaGraphExtractPass : public pir::Pass {
 public:
  CudaGraphExtractPass()
      : pir::Pass("cuda_graph_extract_pass", /*opt_level=*/1) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        common::errors::InvalidArgument(
            "sub_graph_extract_pass should run on module op."));
    auto& block = module_op.block();

    auto IsSupportCudaGraph = [](const pir::Operation& op) {
      return op.name() != "pd_op.attention";
    };

    std::vector<GroupOpsVec> groups =
        ::pir::DetectSubGraphs(&block, IsSupportCudaGraph);

    for (auto& group_ops : groups) {
      VLOG(4) << "current cuda_group count : " << group_ops.size();
      ::pir::ReplaceWithCudaGraphOp(&block, group_ops);
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  void ReplaceWithCudaGraphOp(pir::Block* block, const GroupOpsVec& group_ops) {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
#ifdef PADDLE_WITH_CINN
    ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
#endif
#ifdef PADDLE_WITH_DNNL
    ctx->GetOrRegisterDialect<paddle::dialect::OneDNNOperatorDialect>();
#endif
    ::pir::Builder builder = ::pir::Builder(ctx, block);
    const std::vector<pir::Value> outputs = AnalysisOutputs(group_ops, false);

    // step 1: Analysis and insert group op before insert_point.
    auto* insert_point = FindInsertPoint(group_ops, outputs);
    MoveUpstreamOpBeforeGroup(group_ops, block, insert_point);
    builder.set_insertion_point(insert_point);
    VLOG(6) << "Insert GroupOp after " << insert_point->name();

    // step 2: Replace the old op with CudaGraphOp.
    auto cuda_graph_op = [&]() -> pir::CudaGraphOp {
      std::vector<pir::Type> output_types;
      for (auto& value : outputs) output_types.emplace_back(value.type());

      auto group_op = builder.Build<pir::CudaGraphOp>(output_types);
      for (auto op : group_ops) {
        op->MoveTo(group_op.block(), group_op.block()->end());
      }
      return group_op;
    }();

    // step 3: Replace outputs of inner ops
    const std::vector<pir::Value> group_outs = cuda_graph_op->results();
    std::unordered_set<pir::Operation*> inner_ops(group_ops.begin(),
                                                  group_ops.end());
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].ReplaceUsesWithIf(group_outs[i],
                                   [&inner_ops](pir::OpOperand op) {
                                     return !inner_ops.count(op.owner());
                                   });
    }

    // step 4: Insert YieldOp for outputs
    builder.SetInsertionPointToBlockEnd(new_group_op.block());
    builder.Build<::pir::YieldOp>(outputs);
  }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCudaGraphExtractPass() {
  return std::make_unique<CudaGraphExtractPass>();
}

}  // namespace pir

REGISTER_IR_PASS(cuda_graph_extract_pass, CudaGraphExtractPass);
