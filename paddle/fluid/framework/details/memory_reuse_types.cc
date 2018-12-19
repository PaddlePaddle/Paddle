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

#include "paddle/fluid/framework/details/memory_reuse_types.h"
#include <iostream>
#include <sstream>
#include <string>

namespace paddle {
namespace framework {
namespace details {

size_t NodeSizeInBytes(ir::Node* n) {
  auto* desc = FindVarDescInBlock(n);
  auto shape = desc->GetShape();
  size_t type_size = SizeOfType(desc->GetDataType());
  int size = 1;
  for (auto& s : shape) {
    size *= s;
  }
  return type_size * std::abs(size);
}

std::string DebugStringImpl(VarDesc* var) {
  std::stringstream ss;
  ss << var->Name();
  ss << "[";
  try {
    auto shape = var->GetShape();
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i != shape.size() - 1) {
        ss << shape[i] << ",";
      } else {
        ss << shape[i];
      }
    }
    ss << "]";
  } catch (...) {
    ss << "Var has no VarDesc !!! Name:" << var->Name();
  }
  return ss.str();
}

std::string DebugString(ir::Node* var) {
  return DebugStringImpl(FindVarDescInBlock(var));
}
// return DebugString(var->Var()); }

// NOTE(dzh): based ir node, if a large node has been reused
// by a small size node, then next time it appear in pool, it will
// have the small size. Find the original node shap from blockdesc.
VarDesc* FindVarDescInBlock(ir::Node* n) {
  PADDLE_ENFORCE(n->IsVar() && !n->IsCtrlVar() && n->inputs.size() == 1);
  BlockDesc* block = n->inputs[0]->Op()->Block();
  PADDLE_ENFORCE(block->HasVar(n->Name()),
                 string::Sprintf("Block do not has var %s", n->Name()));
  return block->FindVar(n->Name());
}

struct NodeComparator {
  bool operator()(ir::Node* lhs, ir::Node* rhs) const {
    auto* lhs_desc = FindVarDescInBlock(lhs);
    auto* rhs_desc = FindVarDescInBlock(rhs);
    auto lhs_shape = lhs_desc->GetShape();
    auto rhs_shape = rhs_desc->GetShape();
    if ((lhs_shape[0] == -1 && rhs_shape[0] == -1) ||
        (lhs_shape[0] != -1 && rhs_shape[0] != -1)) {
      return NodeSizeInBytes(lhs) <= NodeSizeInBytes(rhs);
    } else {
      return false;
    }
  }
};

void OrderedNodePairPool::Insert(ir::Node* var, ir::Node* op) {
  PADDLE_ENFORCE(var->IsVar() && !var->IsCtrlVar());
  PADDLE_ENFORCE(op->IsOp());
  if (mark_table_.count(var->Name()) != 0) {
    mark_table_[var->Name()]->second.insert(op);
    return;
  }

  auto* var_desc = FindVarDescInBlock(var);
  auto var_shape = var_desc->GetShape();
  int batch_size = static_cast<int>(var_shape[0]);

  NodeComparator compare_node;
  Iter it = nodes_.begin();
  while (it != nodes_.end()) {
    auto* cache_desc = FindVarDescInBlock(it->first);
    int cache_batch_size = cache_desc->GetShape()[0];
    if ((cache_batch_size == -1 && batch_size == -1) ||
        (cache_batch_size != -1 && batch_size != -1)) {
      if (compare_node(it->first, var)) {
        ++it;
      } else {
        break;
      }
    } else if (cache_batch_size == -1 && batch_size != -1) {
      ++it;
    } else if (cache_batch_size != -1 && batch_size == -1) {
      break;
    }
  }

  it =
      nodes_.insert(it, std::make_pair(var, std::unordered_set<ir::Node*>{op}));
  mark_table_[var->Name()] = it;
}

int OrderedNodePairPool::GetIndex(ir::Node* var) {
  return std::distance(nodes_.begin(), mark_table_[var->Name()]);
}

ir::Node* OrderedNodePairPool::NodeMatch(ir::Node* var) const {
  ir::Node* found_node = nullptr;
  NodeComparator compare_node;

  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    if (compare_node(var, it->first)) {
      found_node = it->first;
      break;
    }
  }
  return found_node;
}

void OrderedNodePairPool::Erase(ir::Node* var) {
  PADDLE_ENFORCE(mark_table_.count(var->Name()));
  nodes_.erase(mark_table_[var->Name()]);
  mark_table_.erase(var->Name());
}

std::string OrderedNodePairPool::ToString() const {
  std::stringstream ss;
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    ss << DebugString(it->first) << " ";
  }
  return ss.str();
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
