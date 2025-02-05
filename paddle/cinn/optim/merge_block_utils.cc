// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/merge_block_utils.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/stmt.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace optim {

namespace {
using ir::stmt::BlockRef;
using ir::stmt::For;
using ir::stmt::StmtRef;

struct ForHash {
  std::size_t operator()(const For& stmt) const {
    return std::hash<const Object*>()(stmt.get());
  }
};

struct ForInfoAnalyzer {
 public:
  void operator()(const For& for_stmt) { Visit(for_stmt); }

  ForTreeNode BuildTreeNode(const For& node) {
    ForTreeNode tree_node = {node, std::vector<ForTreeNode>()};
    for (const For& stmt : for_to_children_[node]) {
      tree_node.children.push_back(BuildTreeNode(stmt));
    }
    return tree_node;
  }

  ForTreeNode GetRootTreeNode() { return BuildTreeNode(root_node_); }

 private:
  void Visit(const For& node) {
    if (root_node_ == nullptr) {
      root_node_ = node;
    }
    const BlockRef& body = node->body();
    for (const StmtRef& stmt : body->stmts()) {
      if (stmt.isa<For>()) {
        for_to_children_[node].push_back(stmt.as<For>());
        Visit(stmt.as<For>());
      }
    }
  }

 private:
  For root_node_{nullptr};
  std::unordered_map<For, std::vector<For>, ForHash> for_to_children_;
};

}  // namespace

bool CanMergeBlocks(const For first,
                    const For second,
                    const ForEqualFunc& IsEqual) {
  auto Get = [&](const For for_stmt) -> ForTreeNode {
    ForInfoAnalyzer for_info_analyzer;
    for_info_analyzer(for_stmt);
    return for_info_analyzer.GetRootTreeNode();
  };
  const auto first_inner_for_list = Get(first);
  const auto second_inner_for_list = Get(second);
  return IsEqual(first_inner_for_list, second_inner_for_list);
}

}  // namespace optim
}  // namespace cinn
