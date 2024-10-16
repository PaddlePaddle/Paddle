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
#include "paddle/common/enforce.h"

namespace cinn {
namespace optim {

namespace {

struct ForInfoAnalyzer : public ir::IRMutator<Expr*> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  ForTreeNode BuildTreeNode(const ir::For* node) {
    ForTreeNode tree_node = {node, std::vector<ForTreeNode>()};
    for (const auto for_node : for_to_children_[node]) {
      tree_node.children.push_back(BuildTreeNode(for_node));
    }
    return tree_node;
  }

  ForTreeNode GetRootTreeNode() { return BuildTreeNode(root_node_); }

 private:
  void Visit(const ir::For* node, ir::Expr* expr) override {
    auto old_last_node = last_node_;
    if (last_node_ == nullptr) {
      root_node_ = node;
    } else {
      for_to_children_[last_node_].push_back(node);
    }
    last_node_ = const_cast<ir::For*>(node);
    ir::IRMutator<>::Visit(node, expr);
    last_node_ = old_last_node;
  }

  ir::For* last_node_ = nullptr;
  const ir::For* root_node_ = nullptr;
  std::unordered_map<const ir::For*, std::vector<const ir::For*>>
      for_to_children_;
};

}  // namespace

bool CanMergeBlocks(const ir::For* first,
                    const ir::For* second,
                    const ForEqualFunc& IsEqual) {
  auto Get = [&](ir::Expr* expr) -> ForTreeNode {
    ForInfoAnalyzer for_info_analyzer;
    for_info_analyzer(expr);
    return for_info_analyzer.GetRootTreeNode();
  };
  ir::Expr first_expr = Expr(const_cast<ir::For*>(first));
  ir::Expr second_expr = Expr(const_cast<ir::For*>(second));
  const auto first_inner_for_list = Get(&first_expr);
  const auto second_inner_for_list = Get(&second_expr);
  return IsEqual(first_inner_for_list, second_inner_for_list);
}

}  // namespace optim
}  // namespace cinn
