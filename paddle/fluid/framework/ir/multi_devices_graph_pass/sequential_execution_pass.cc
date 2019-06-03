// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimize_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

static bool IsSameOpDesc(OpDesc *op1, OpDesc *op2) {
  return op1->Type() == op2->Type() && op1->Inputs() == op2->Inputs() &&
         op1->Outputs() == op2->Outputs();
}

class SequentialExecutionPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override {
    // FIXME(zjl): Insert dependencies between some distributed ops may cause
    // the multi_devices_graph_pass fails. So we skip these ops here.
    // Indeed, maybe we should not insert dependencies between these ops
    // casually, which may cause deadlock easily.
    // We should add more skipped distributed ops when found errors in
    // multi_devices_graph_pass
    static std::unordered_set<std::string> skip_dist_ops{
        "send", "recv", "send_barrier", "fetch_barrier"};

    auto &ops =
        graph->Get<const std::vector<OpDesc *>>(details::kStaleProgramOpDescs);
    std::vector<ir::Node *> op_node_list;
    op_node_list.reserve(ops.size());

    std::unordered_map<ir::Node *, size_t> op_deps;
    std::unordered_map<ir::Node *, std::unordered_set<ir::Node *>> pending_ops;
    std::unordered_set<ir::Node *> ready_ops;

    for (ir::Node *node : graph->Nodes()) {
      if (!node->IsOp()) continue;
      std::unordered_set<ir::Node *> preceding_ops;
      for (auto *in : node->inputs) {
        PADDLE_ENFORCE(in->IsVar(),
                       "Preceding Node of Op Nodes must be Var Node");
        if (in->inputs.empty()) continue;
        PADDLE_ENFORCE(in->inputs.size() == 1 && in->inputs[0]->IsOp(),
                       "Preceding Op Node of Var Node must be unique");
        preceding_ops.insert(in->inputs[0]);
        pending_ops[in->inputs[0]].insert(node);
      }
      op_deps[node] = preceding_ops.size();
      if (preceding_ops.empty()) {
        ready_ops.insert(node);
      }
    }

    for (auto *op_desc : ops) {
      ir::Node *found_node = nullptr;
      for (auto *node : ready_ops) {
        if (IsSameOpDesc(op_desc, node->Op())) {
          PADDLE_ENFORCE(found_node == nullptr,
                         "Found multiple op_desc in graph: %s",
                         op_desc->Type());
          found_node = node;
        }
      }

      PADDLE_ENFORCE_NOT_NULL(found_node, "Cannot find op_desc in graph: %s",
                              op_desc->Type());
      for (auto *pending_op : pending_ops[found_node]) {
        if (--op_deps.at(pending_op) == 0) {
          ready_ops.insert(pending_op);
        }
      }
      ready_ops.erase(found_node);
      if (skip_dist_ops.count(op_desc->Type()) == 0) {
        op_node_list.push_back(found_node);
      }
    }

    for (size_t i = 1; i < op_node_list.size(); ++i) {
      auto *dep_var = graph->CreateControlDepVar();
      op_node_list[i]->inputs.push_back(dep_var);
      op_node_list[i - 1]->outputs.push_back(dep_var);
      dep_var->outputs.push_back(op_node_list[i]);
      dep_var->inputs.push_back(op_node_list[i - 1]);
      VLOG(10) << "Add dependencies between " << op_node_list[i - 1]->Name()
               << " and " << op_node_list[i]->Name();
    }
  }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(sequential_execution_pass,
              paddle::framework::ir::SequentialExecutionPass)
    .RequireGraphAttr(paddle::framework::details::kStaleProgramOpDescs);
