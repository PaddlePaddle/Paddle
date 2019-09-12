/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/group.h"
#include "paddle/fluid/framework/ir/node.h"
namespace paddle {
namespace framework {
namespace ir {

bool Group::ExistInGroup(Node* node) {
  bool ret = true;
  if (fused_node_.find(node) == fused_node_.end()) {
    ret = false;
  }
  return ret;
}

void Group::PrintGroup() {
  for (auto node : fused_node_) std::cout << node->id() << std::endl;
}

void ElementwiseGroupDetector::MarkFusedGroup(const Graph& graph) {
  for (auto& node : GraphTraits::DFS(graph)) {
    auto output_nodes = node.outputs;
    auto input_nodes = node.inputs;

    auto elementwise_node_boolean = [](Node* node) -> bool {
      return node->NodeType() == ir::Node::Type::kOperation &&
             elementwise_op.find(node->Op()->Type()) != elementwise_op.end();
    };
    if (elementwise_node_boolean(&node)) {
      std::vector<Node*> adjacent_op{};

      for (size_t i = 0; i < input_nodes.size(); i++) {
        auto pre_ops = input_nodes[i]->inputs;
        for (size_t j = 0; j < pre_ops.size(); j++) {
          if (elementwise_node_boolean(pre_ops[j])) {
            adjacent_op.push_back(pre_ops[j]);
          }
        }
      }

      for (size_t i = 0; i < output_nodes.size(); i++) {
        auto next_ops = output_nodes[i]->inputs;
        for (size_t j = 0; j < next_ops.size(); j++) {
          if (elementwise_node_boolean(next_ops[j])) {
            adjacent_op.push_back(next_ops[j]);
          }
        }
      }

      bool exist_group = false;
      for (size_t i = 0; i < adjacent_op.size(); i++) {
        for (size_t j = 0; j < groups.size(); j++) {
          exist_group = (groups[j].ExistInGroup(adjacent_op[i]) ||
                         groups[j].ExistInGroup(&node));
          if (exist_group) {
            groups[j].InsertNode(&node);
            break;
          }
        }
      }
      if (!exist_group) {
        Group tmp_group;
        for (size_t j = 0; j < adjacent_op.size(); j++) {
          tmp_group.InsertNode(adjacent_op[j]);
        }
        tmp_group.InsertNode(&node);
        groups.push_back(tmp_group);
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
