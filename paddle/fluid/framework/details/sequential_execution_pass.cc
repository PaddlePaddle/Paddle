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

#include "paddle/fluid/framework/details/sequential_execution_pass.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {
namespace details {

static bool IsSameOpDesc(OpDesc *op1, OpDesc *op2) {
  return op1->Type() == op2->Type() && op1->Inputs() == op2->Inputs() &&
         op1->Outputs() == op2->Outputs();
}

std::unique_ptr<ir::Graph> SequentialExecutionPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto ops = this->Get<const std::vector<OpDesc *>>(kAllOpDescs);
  std::vector<ir::Node *> op_node_list;
  op_node_list.reserve(ops.size());

  std::unordered_map<ir::Node *, size_t> op_deps;
  std::unordered_map<ir::Node *, std::unordered_set<ir::Node *>> pending_ops;
  std::unordered_set<ir::Node *> ready_ops;

  for (ir::Node *node : graph->Nodes()) {
    if (!node->IsOp()) continue;
    std::unordered_set<ir::Node *> preceding_ops;
    pending_ops[node];
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
                       "Found multiple op_desc in graph: %s", op_desc->Type());
        found_node = node;
      }
    }

    PADDLE_ENFORCE_NOT_NULL(found_node, "Cannot find op_desc in graph: %s",
                            found_node->Op()->Type());
    for (auto *pending_op : pending_ops.at(found_node)) {
      if (--op_deps.at(pending_op) == 0) {
        ready_ops.insert(pending_op);
      }
    }
    ready_ops.erase(found_node);
    op_node_list.push_back(found_node);
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
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(sequential_execution_pass,
              paddle::framework::details::SequentialExecutionPass)
    .RequirePassAttr(paddle::framework::details::kAllOpDescs);
