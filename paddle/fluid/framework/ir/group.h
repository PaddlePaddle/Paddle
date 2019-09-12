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

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace framework {
namespace ir {

static std::set<std::string> elementwise_op = {"elementwise_add",
                                               "elementwise_div",
                                               "elementwise_sub",
                                               "elementwise_mul",
                                               "relu",
                                               "sigmoid"};
class Group {
 public:
  size_t size() { return fused_node_.size(); }
  void InsertRootId(int id) { root_id_.push_back(id); }
  std::vector<int> GetRootId() { return root_id_; }
  void InsertNode(Node* node) { fused_node_.insert(node); }
  bool ExistInGroup(Node* node);
  void PrintGroup();

 private:
  std::unordered_set<Node*> fused_node_;
  std::vector<int> root_id_;
};

class ElementwiseGroupDetector {
 public:
  void MarkFusedGroup(const Graph& graph);
  std::vector<Group> GetFusedGroups() { return groups; }
  void ClearGraphGroup() { groups.clear(); }

 private:
  std::vector<Group> groups;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
