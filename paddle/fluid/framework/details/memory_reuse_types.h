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
#include <iterator>
#include <list>
#include <string>
#include <vector>
#include <utility>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {

class OrderedReusedNodePairPool {
  // O(1) insert, delete. sorted by node size.
 public:
  using Iter = typename std::list<std::pair<ir::Node*, ir::Node*>>::iterator;
  using ConstIter =
      typename std::list<std::pair<ir::Node*, ir::Node*>>::const_iterator;
  void Insert(ir::Node* var, ir::Node* op);
  void Erase(ir::Node* var);
  bool Has(ir::Node* var) { return mark_table_.count(var->Name()); }
  Iter begin() { return nodes_.begin(); }
  Iter end() { return nodes_.end(); }
  ConstIter begin() const { return nodes_.begin(); }
  ConstIter end() const { return nodes_.end(); }
  size_t size() const { return nodes_.size(); }

 private:
  // for searching.
  std::unordered_map<std::string, Iter> mark_table_;
  // node swap pairs. var -> cache var
  std::list<std::pair<ir::Node*, ir::Node*>> nodes_;
};

constexpr char kFetchedVars[] = "fetched_vars";
constexpr char kUnlivedNodePool[] = "unused_node_pool";
constexpr char kReusedNodePairMap[] = "reused_nodepair_map";
constexpr char kGraphOpsReused[] = "graph_ops_reused";
constexpr char kGraphEarlyDeleteOpsDeps[] = "graph_early_delete_ops_deps";

using UnlivedNodePool = std::vector<
    std::pair<std::string,                                  /*var node*/
              int> /*the last op which use var node id*/>;  // order matters
using ReusedNodePairMap = std::unordered_map<
    int /*op order id*/,
    std::pair<std::string /*var*/, std::string /*reused var*/>>;
using GraphOpsReused = std::vector<int /*op order id*/>;
using GraphEarlyDeleteOpsDeps = std::vector<std::vector<int /*op order id*/>>;
// node memory size in bytes
inline static size_t GetNodeSize(ir::Node* n);

}  // namespace details

// implement
namespace details {
inline static size_t GetNodeSize(ir::Node* n) {
  auto* desc = n->Var();
  auto shape = desc->GetShape();
  size_t type_size =
      framework::SizeOfType(framework::ToTypeIndex(desc->GetDataType()));
  int size = 1;
  for (auto& s : shape) {
    size *= s;
  }
  return type_size * std::abs(size);
}

inline void OrderedReusedNodePairPool::Insert(ir::Node* var, ir::Node* op) {
  using NodePair = std::pair<ir::Node*, ir::Node*>;
  auto var_bytes_comparator = [](const NodePair& lhs, ir::Node* rhs) {
    auto lhs_size = GetNodeSize(lhs.first);
    auto rhs_size = GetNodeSize(rhs);
    if (lhs_size == rhs_size) {
      // -1 means batch_size, so [-1, 1,...] > [1, ...] when their abs value
      // equal.
      auto* lhs_desc = lhs.first->Var();
      auto lhs_shape = lhs_desc->GetShape();
      return lhs_shape[0] != -1;
    } else {
      return lhs_size < rhs_size;
    }
  };

  PADDLE_ENFORCE(var->IsVar() && !var->IsCtrlVar());
  PADDLE_ENFORCE(op->IsOp());
  PADDLE_ENFORCE(mark_table_.count(var->Name()) == 0);
  Iter it =
      std::lower_bound(nodes_.begin(), nodes_.end(), var, var_bytes_comparator);
  it = nodes_.insert(it, std::make_pair(var, op));
  mark_table_[var->Name()] = it;
}

inline void OrderedReusedNodePairPool::Erase(ir::Node* var) {
  PADDLE_ENFORCE(mark_table_.count(var->Name()));
  nodes_.erase(mark_table_[var->Name()]);
  mark_table_.erase(var->Name());
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
