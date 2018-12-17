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

#pragma once
#include <algorithm>
#include <iostream>
#include <iterator>
#include <list>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {

constexpr char kFetchedVars[] = "fetched_vars";
constexpr char kGraphNodePool[] = "graph_node_pool";

// NOTE(dzh): Variable and the operators use the var.
// for early delete pass.
// Because analysis var pass build base on ir::Node, which maybe released
// or modified between passes, so we use OpDesc* to mark ops.
using GraphNodePool = std::vector<
    std::pair<std::string /*var node*/, std::unordered_set<OpDesc*> /* ops */>>;

// NOTE(dzh): by default, it sort node in ascend order(by node bytes size).
// in fluid, -1 means the batch_size is determined in runtime.
// the node batch_size equal -1 always ranking in the front than the node not.
// For example,
// node0[-1, 1] node1[-1, 1, 1], node2[1,1], node3[1,1024], ..
// O(1) insert, delete
class OrderedNodePairPool {
 public:
  using NodePair = std::pair<ir::Node*, std::unordered_set<ir::Node*>>;
  using Iter = typename std::list<NodePair>::iterator;
  using ConstIter = typename std::list<NodePair>::const_iterator;

  void Insert(ir::Node* var, ir::Node* op);

  void Erase(ir::Node* var);

  bool Has(ir::Node* var) { return mark_table_.count(var->Name()); }

  ir::Node* NodeMatch(ir::Node* var) const;
  // map store non-const iterator, can not promise const
  int GetIndex(ir::Node* var);
  // pool all node to string
  std::string ToString() const;

  Iter begin() { return nodes_.begin(); }
  Iter end() { return nodes_.end(); }
  ConstIter begin() const { return nodes_.begin(); }
  ConstIter end() const { return nodes_.end(); }
  size_t size() const { return nodes_.size(); }

 private:
  // for searching.
  std::unordered_map<std::string, Iter> mark_table_;
  // node swap pairs. var -> ops dep var
  std::list<NodePair> nodes_;
};

// node memory size in bytes
size_t NodeSizeInBytes(ir::Node* n);

std::string DebugString(ir::Node* var);

// std::string DebugString(VarDesc* var);
VarDesc* FindVarDescInBlock(ir::Node* n);

}  // namespace details
}  // namespace framework
}  // namespace paddle
