// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/operators/controlflow/conditional_block_op_helper.h"
#include "paddle/fluid/operators/controlflow/op_variant.h"
namespace paddle::framework::ir {
using OpVariant = operators::OpVariant;
class ConditionalOpEagerDeletionPass : public Pass {
 protected:
  void ApplyImpl(Graph *graph) const override {
    auto all_ops = ir::FilterByNodeWrapper<details::OpHandleBase>(*graph);

    // Find all conditional_op and conditional_grad_op
    std::unordered_map<
        size_t,
        std::pair<std::vector<OpVariant>, std::vector<OpVariant>>>
        target_ops;
    for (auto *op : all_ops) {
      auto compute_op = dynamic_cast<details::ComputationOpHandle *>(op);
      if (compute_op == nullptr) continue;

      if (compute_op->Name() == "conditional_block") {
        target_ops[compute_op->GetScopeIdx()].first.emplace_back(
            compute_op->GetOp());
      } else if (compute_op->Name() == "conditional_block_grad") {
        target_ops[compute_op->GetScopeIdx()].second.emplace_back(
            compute_op->GetOp());
      }
    }

    // NOTE(Aurelius84): In case of @to_static, after we finish executing
    // forward graph, some necessary variable in step_scope of controlflow_op
    // should be kept for backward graph.
    if (graph->IsConstructedByPartialProgram()) {
      PADDLE_ENFORCE_LE(target_ops.size(),
                        1,
                        common::errors::InvalidArgument(
                            "Unsupported multi devices if graph is constructed "
                            "with partial program."));
      size_t scope_idx = 0;
      auto &ifelse_ops = target_ops[scope_idx].first;
      auto &ifelse_grad_ops = target_ops[scope_idx].second;

      auto all_ops = graph->OriginProgram().Block(0).AllOps();
      if (ifelse_ops.empty()) {
        operators::AppendOpVariantByOpName(
            all_ops, std::string("conditional_block"), &ifelse_ops);
      } else if (ifelse_grad_ops.empty()) {
        operators::AppendOpVariantByOpName(
            all_ops, std::string("conditional_block_grad"), &ifelse_grad_ops);
      } else {
        PADDLE_THROW("One of ifelse_ops or ifelse_grad_ops should be empty.");
      }
    }

    for (auto &ops_pair : target_ops) {
      auto &ifelse_ops = ops_pair.second.first;
      auto &ifelse_grad_ops = ops_pair.second.second;
      operators::PrepareSafeEagerDeletionOnConditionalOpAndConditionalGradOp(
          graph->OriginProgram(), ifelse_ops, ifelse_grad_ops);
    }

    for (auto op_handler : all_ops) {
      auto *compute_op =
          dynamic_cast<details::ComputationOpHandle *>(op_handler);
      if (compute_op == nullptr) continue;
      if (compute_op->Name() == "conditional_block" ||
          compute_op->Name() == "conditional_block_grad") {
        ir::Node *op_node = op_handler->Node();
        auto *op_base = compute_op->GetOp();
        if (op_base->Attrs().count("skip_eager_deletion_vars")) {
          op_node->Op()->SetAttr(
              "skip_eager_deletion_vars",
              op_base->Attrs().at("skip_eager_deletion_vars"));
        }
      }
    }
  }
};

}  // namespace paddle::framework::ir

REGISTER_PASS(conditional_block_op_eager_deletion_pass,
              paddle::framework::ir::ConditionalOpEagerDeletionPass);
