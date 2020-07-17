// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <queue>
#include <unordered_set>
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class AddReaderDependencyPass : public Pass {
 protected:
  void ApplyImpl(Graph *graph) const override;
};

static std::unordered_set<Node *> FindAllPrecedingOpNodes(Node *node) {
  std::unordered_set<Node *> result;

  std::queue<Node *> q;
  q.push(node);

  while (!q.empty()) {
    auto *cur_node = q.front();
    q.pop();

    for (auto &in_var : cur_node->inputs) {
      for (auto &in_op : in_var->inputs) {
        if (result.count(in_op) == 0 && in_op != node) {
          result.insert(in_op);
          q.push(in_op);
        }
      }
    }
  }
  return result;
}

void AddReaderDependencyPass::ApplyImpl(Graph *graph) const {
  const auto &nodes = graph->Nodes();
  std::unordered_set<Node *> ops;
  std::unordered_set<Node *> read_ops;

  for (auto &n : nodes) {
    if (n->IsOp() && n->Op()) {
      ops.insert(n);

      if (n->Op()->Type() == "read") {
        read_ops.insert(n);
      }
    }
  }

  VLOG(10) << "Found " << read_ops.size() << " read op(s)";

  if (read_ops.empty()) {
    return;
  }

  // Find all startup ops
  std::unordered_set<Node *> out_ops;
  for (auto &op : ops) {
    for (auto &out_var : op->outputs) {
      for (auto &out_op : out_var->outputs) {
        out_ops.insert(out_op);
      }
    }
  }

  for (auto &out_op : out_ops) {
    ops.erase(out_op);
  }

  VLOG(10) << "Found " << ops.size() << " startup ops";

  for (auto &read_op : read_ops) {
    auto preceding_ops = FindAllPrecedingOpNodes(read_op);
    for (auto &startup_op : ops) {
      if (read_op == startup_op || preceding_ops.count(startup_op) > 0) {
        VLOG(10) << "Startup op " << startup_op->Op()->Type() << " is skipped";
        continue;
      }

      auto *dep_var = graph->CreateControlDepVar();
      read_op->outputs.push_back(dep_var);
      startup_op->inputs.push_back(dep_var);
      dep_var->inputs.push_back(read_op);
      dep_var->outputs.push_back(startup_op);

      VLOG(10) << "Add dependencies between " << read_op->Op()->Type()
               << " and " << startup_op->Op()->Type();
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(add_reader_dependency_pass,
              paddle::framework::ir::AddReaderDependencyPass);
