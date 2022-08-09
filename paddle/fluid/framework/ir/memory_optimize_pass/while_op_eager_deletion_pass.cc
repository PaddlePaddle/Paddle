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
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/operators/controlflow/op_variant.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"

namespace paddle {
namespace framework {
namespace ir {
using OpVariant = operators::OpVariant;

class WhileOpEagerDeletionPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override {
    if (!graph->IsMainGraph()) {
      // TODO(zhhsplendid): the WhileOpEagerDeletionPass is based on old Graph,
      // which only applies to the main block graph. The new Eager Deletion
      // Technical can be added after we write new while_op based on SubGraph
      // instead of SubBlock
      return;
    }
    auto all_ops = ir::FilterByNodeWrapper<details::OpHandleBase>(*graph);

    // Find all while_op and while_grad_op. In case of @to_static, graph
    // may be constructed only by forward or backward program, so we use
    // OpVariant here instead of OperatorBase.
    std::unordered_map<
        size_t,
        std::pair<std::vector<OpVariant>, std::vector<OpVariant>>>
        target_ops;
    for (auto *op : all_ops) {
      auto compute_op = dynamic_cast<details::ComputationOpHandle *>(op);
      if (compute_op == nullptr) continue;

      if (compute_op->Name() == "while") {
        target_ops[compute_op->GetScopeIdx()].first.emplace_back(
            compute_op->GetOp());
      } else if (compute_op->Name() == "while_grad") {
        target_ops[compute_op->GetScopeIdx()].second.emplace_back(
            compute_op->GetOp());
      }
    }
    if (graph->IsConstructedByPartialProgram()) {
      VLOG(4) << "Is Paritial Program";
      PADDLE_ENFORCE_LE(
          target_ops.size(),
          1,
          platform::errors::InvalidArgument(
              "Unsupported multi device if graph is constructed by "
              "partial program."));
      size_t scope_idx = 0;
      auto &while_ops = target_ops[scope_idx].first;
      auto &while_grad_ops = target_ops[scope_idx].second;

      auto all_ops = graph->OriginProgram().Block(0).AllOps();
      if (while_ops.empty()) {
        VLOG(1) << "AppendOpVariantByOpName: while";
        operators::AppendOpVariantByOpName(
            all_ops, std::string("while"), &while_ops);
      } else if (while_grad_ops.empty()) {
        VLOG(1) << "AppendOpVariantByOpName: while_grad";
        operators::AppendOpVariantByOpName(
            all_ops, std::string("while_grad"), &while_grad_ops);
      } else {
        PADDLE_THROW("One of while_ops or while_grad_ops should be empty.");
      }
    }

    for (auto &ops_pair : target_ops) {
      VLOG(4) << "Scope Idx = " << ops_pair.first;
      auto &while_ops = ops_pair.second.first;
      VLOG(4) << "while_ops.size() = " << while_ops.size();
      auto &while_grad_ops = ops_pair.second.second;
      VLOG(4) << "while_grad_ops.size() = " << while_grad_ops.size();
      operators::PrepareSafeEagerDeletionOnWhileOpAndWhileGradOp(
          graph->OriginProgram(), while_ops, while_grad_ops);
    }

    // VLOG(1) << "======>end eager delete while op";
    // for (auto *op : all_ops) {
    //   auto compute_op = dynamic_cast<details::ComputationOpHandle *>(op);
    //   if (compute_op == nullptr) continue;

    //   if (compute_op->Name() == "while") {
    //     VLOG(1) << "get nodes: while";
    //     auto* while_op_base = compute_op->GetOp();

    //     std::string attr_str = "(";
    //     std::vector<std::string> skip_eager_deletion_vars;
    //     if (while_op_base->Attrs().count("skip_eager_deletion_vars")) {
    //       skip_eager_deletion_vars =
    //       PADDLE_GET_CONST(std::vector<std::string>,
    //       while_op_base->Attrs().at("skip_eager_deletion_vars"));
    //       attr_str.append("skip_eager_deletion_vars(");
    //       for (size_t j=0; j < skip_eager_deletion_vars.size(); j++) {
    //         attr_str.append(skip_eager_deletion_vars[j]);
    //         attr_str.append(",");
    //       }
    //     }
    //     attr_str.append(")");
    //     VLOG(1) << "while node attr skip_eager_deletion_vars: " << attr_str;

    //   } else if (compute_op->Name() == "while_grad") {
    //     VLOG(1) << "get nodes: while_grad";
    //     auto* while_grad_op_base = compute_op->GetOp();

    //     std::string attr_str = "(";
    //     std::vector<std::string> skip_eager_deletion_vars;
    //     if (while_grad_op_base->Attrs().count("skip_eager_deletion_vars")) {
    //       skip_eager_deletion_vars =
    //       PADDLE_GET_CONST(std::vector<std::string>,
    //       while_grad_op_base->Attrs().at("skip_eager_deletion_vars"));
    //       attr_str.append("skip_eager_deletion_vars(");
    //       for (size_t j=0; j < skip_eager_deletion_vars.size(); j++) {
    //         attr_str.append(skip_eager_deletion_vars[j]);
    //         attr_str.append(",");
    //       }
    //     }
    //     attr_str.append(")");
    //     VLOG(1) << "while node attr skip_eager_deletion_vars: " << attr_str;
    //   }
    // }
    // VLOG(1) <<  "get all from graph node ";
    // std::vector<OpDesc> ops;
    // for (auto* n : graph->Nodes()) {
    //   // if node is not Op, skip
    //   if (!n->IsOp()) continue;
    //   VLOG(1) << "Node : " << n->Name();
    //   if (n->Name() == "while" || n->Name() == "while_grad") {
    //     auto* op_desc = n->Op();

    //     std::string attr_str = "(";
    //     std::vector<std::string> skip_eager_deletion_vars;
    //     if (op_desc->GetAttrMap().count("skip_eager_deletion_vars")) {
    //       skip_eager_deletion_vars =
    //       PADDLE_GET_CONST(std::vector<std::string>,
    //       op_desc->GetAttrMap().at("skip_eager_deletion_vars"));
    //       attr_str.append("skip_eager_deletion_vars(");
    //       for (size_t j=0; j < skip_eager_deletion_vars.size(); j++) {
    //         attr_str.append(skip_eager_deletion_vars[j]);
    //         attr_str.append(",");
    //       }
    //     }
    //     attr_str.append(")");
    //     VLOG(1) << "while node attr skip_eager_deletion_vars: " << attr_str;
    //   }
    // }
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(while_op_eager_deletion_pass,
              paddle::framework::ir::WhileOpEagerDeletionPass);
