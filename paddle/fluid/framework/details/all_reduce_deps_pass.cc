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
    const std::unordered_set<ir::Node *> &nodes) {
  std::unique_ptr<std::unordered_set<ir::Node *>> all_reduce_nodes(
      new std::unordered_set<ir::Node *>());
  for (auto &node : nodes) {
    if (!node->IsOp()) {
      continue;
    }
    if (node->Name() != "all_reduce" && node->Name() != "allreduce") {
      continue;
    }
    // std::cout << node->Name() << std::endl;
    all_reduce_nodes->insert(node);
  }
  return all_reduce_nodes;
}

std::unique_ptr<ir::Graph> AllReduceDepsPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  /*
auto &ops = Get<const std::vector<OpDesc *>>(kAllOpDescs);
for (auto *op : ops) {
    auto& inputs = op->Inputs();
    auto& outputs = op->Outputs();
    std::cout << "op name:" << op->Type() << std::endl;
    for (auto& t:inputs){
        std::cout << "\tinput name: " << t.first << ", value:";
        for(auto& v:t.second){
          std::cout  << v << ",";
        }
        std::cout << std::endl;
    }
    for (auto& t:outputs){
        std::cout << "\toutput name: " << t.first << ", value:";
        for(auto& v:t.second){
          std::cout << v << ",";
        }
        std::cout << std::endl;
    }
}
auto allreduce_nodes = FindAllReduceNodes(graph->Nodes());
for(ir::Node* node:*allreduce_nodes.get()){
    std::cout << "allreduce node name:" << node->Name() <<std::endl;
        // << ", opdesc:" << node->Op()->Type() << std::endl;
    for(auto& t:node->inputs){
        if (t->IsVar()){
            std::cout << "\tinput var name:" << t->Name() << std::endl;
        }
    }

    for(auto& t:node->outputs){
        if (t->IsVar()){
            std::cout << "\toutput var name:" << t->Name() << std::endl;
        }

    }
}

for (size_t i = 1; i < node_list.size(); ++i) {
  auto *dep_var = graph->CreateControlDepVar();
  node_list[i]->inputs.push_back(dep_var);
  node_list[i - 1]->outputs.push_back(dep_var);
  dep_var->outputs.push_back(node_list[i]);
  dep_var->inputs.push_back(node_list[i - 1]);
  VLOG(10) << "Add all_reduce Sequential dependencies between "
           << node_list[i - 1]->Name() << " and " << node_list[i]->Name();
}
*/

  auto &graph_ops = graph->Get<GraphOps>(kGraphOps);
  OpGraphView graph_view(all_ops);
  std::unordered_set<OpHandleBase *> cur_level_ops;
  std::vector<std::unordered_set<AllReduceOpHandle *>> all_reduce_ops;

  for (auto *op : graph_vew.AllOps()) {
  }

  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(all_reduce_deps_pass,
              paddle::framework::details::AllReduceDepsPass)
    .RequirePassAttr(paddle::framework::details::kAllOpDescs);
