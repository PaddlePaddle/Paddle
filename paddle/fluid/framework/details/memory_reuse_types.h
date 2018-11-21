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
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {

// NOTE(dzh): by default, it sort node in ascend order(by node bytes size).
// in fluid, -1 means the batch_size is determined in runtime. so we always
// put the node batch_size equal -1 front than the node not.
// For example,
// node0[-1, 1] node1[-1, 1, 1], node2[1,1], node3[1,1024], ..
// O(1) insert, delete
class OrderedReusedNodePairPool {
 public:
  using Iter = typename std::list<std::pair<ir::Node*, ir::Node*>>::iterator;
  using ConstIter =
      typename std::list<std::pair<ir::Node*, ir::Node*>>::const_iterator;

  void Insert(ir::Node* var, ir::Node* op);

  void Erase(ir::Node* var);

  bool Has(ir::Node* var) { return mark_table_.count(var->Name()); }

  ir::Node* NodeMatch(ir::Node* var) const;
  // map store non-const iterator, can not promise const
  int GetPosition(ir::Node* var);

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

struct NodeComparator {
  bool operator()(ir::Node* lhs, ir::Node* rhs) const {
    auto* lhs_desc = lhs->Var();
    auto* rhs_desc = rhs->Var();
    auto lhs_shape = lhs_desc->GetShape();
    auto rhs_shape = rhs_desc->GetShape();

    if ((lhs_shape[0] == -1 && rhs_shape[0] == -1) ||
        (lhs_shape[0] != -1 && rhs_shape[0] != -1)) {
      // NOTE(dzh): dynamic batch size node can not
      // be replaced by static batch size node.
      return GetNodeSize(lhs) <= GetNodeSize(rhs);
    } else {
      return false;
    }
  }
};

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
  PADDLE_ENFORCE(var->IsVar() && !var->IsCtrlVar());
  PADDLE_ENFORCE(op->IsOp());
  PADDLE_ENFORCE(mark_table_.count(var->Name()) == 0);

  auto var_shape = var->Var()->GetShape();
  int batch_size = static_cast<int>(var_shape[0]);
  NodeComparator compare_node;

  Iter it = nodes_.begin();
  while (it != nodes_.end()) {
    int cache_batch_size = it->first->Var()->GetShape()[0];
    if ((cache_batch_size == -1 && batch_size == -1) ||
        (cache_batch_size != -1 && batch_size != -1)) {
      if (!compare_node(var, it->first)) {
        ++it;
      } else {
        break;
      }
    } else if (cache_batch_size == -1 && batch_size != -1) {
      ++it;
    } else if (cache_batch_size != -1 && batch_size == -1) {
      break;
    }
    // if (it) {
    // }
    // if (compare_node(var, it->first)) {
    //   ++it;
    // } else {
    //   // put the node batch_size != -1 at last position
    //   if (batch_size != -1 && it->first->Var()->GetShape()[0] == -1) {
    //     ++it;
    //   } else {
    //     break;
    //   }
    // }
  }
  it = nodes_.insert(it, std::make_pair(var, op));
  mark_table_[var->Name()] = it;
}

inline int OrderedReusedNodePairPool::GetPosition(ir::Node* var) {
  return std::distance(nodes_.begin(), mark_table_[var->Name()]);
}

inline ir::Node* OrderedReusedNodePairPool::NodeMatch(ir::Node* var) const {
  // auto compare_node_size = [&](ir::Node* lhs, ir::Node* rhs) {
  //   // return GetNodeSize(lhs) <= GetNodeSize(rhs);
  // };
  ir::Node* found_node = nullptr;

  // linear search in an sorted node set, find the best fit node.
  NodeComparator compare_node;
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    if (compare_node(var, it->first)) {
      found_node = it->first;
      break;
    }
  }
  return found_node;
}

inline void OrderedReusedNodePairPool::Erase(ir::Node* var) {
  PADDLE_ENFORCE(mark_table_.count(var->Name()));
  nodes_.erase(mark_table_[var->Name()]);
  mark_table_.erase(var->Name());
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
