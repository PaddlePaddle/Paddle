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

#include "paddle/fluid/framework/details/all_reduce_deps_pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace details {

static std::unique_ptr<std::unordered_set<ir::Node *>> FindAllReduceNodes(
    const std::vector<ir::Node *> &nodes) {
  std::unique_ptr<std::unordered_set<ir::Node *>> all_reduce_nodes(
      new std::unordered_set<ir::Node *>());
  for (auto &node : nodes) {
    if (!node->IsOp()) {
      continue;
    }
    if (node->Name() != "all_reduce") {
      continue;
    }
    all_reduce_nodes->insert(node);
  }
  return all_reduce_nodes;
}

std::unique_ptr<ir::Graph> AllReduceDepsPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto node_list = FindAllReduceNodes(graph->Nodes());

  for (size_t i = 1; i < node_list.size(); ++i) {
    auto *dep_var = graph->CreateControlDepVar();
    node_list[i]->inputs.push_back(dep_var);
    node_list[i - 1]->outputs.push_back(dep_var);
    dep_var->outputs.push_back(node_list[i]);
    dep_var->inputs.push_back(node_list[i - 1]);
    VLOG(10) << "Add all_reduce Sequential dependencies between "
             << node_list[i - 1]->Name() << " and " << node_list[i]->Name();
  }

  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(all_reduce_deps_pass,
              paddle::framework::details::AllReduceDepsPass)
