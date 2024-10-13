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
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
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

struct ForVarExtent {
  ir::Var loop_var;
  ir::Expr extent;
};

ir::Expr ReplaceSbrIter(const ir::ScheduleBlockRealize* sbr,
                        const std::vector<ForVarExtent>& src_vars,
                        const std::vector<ForVarExtent>& dst_vars) {
  auto ConstructForVarReplaceMap =
      [&](const std::vector<ForVarExtent>& lhs_extents,
          const std::vector<ForVarExtent>& rhs_extents)
      -> std::unordered_map<ir::Var, ir::Var> {
    std::unordered_map<ir::Var, ir::Var> ret;
    std::unordered_set<std::size_t> visited_rhs_index;
    for (const auto& [lhs_var, lhs_extent] : lhs_extents) {
      for (std::size_t i = 0; i < rhs_extents.size(); ++i) {
        const auto& [rhs_var, rhs_extent] = rhs_extents[i];
        if (cinn::common::AutoSimplify(ir::Sub::Make(lhs_extent, rhs_extent)) ==
                ir::Expr(0) &&
            visited_rhs_index.count(i) == 0) {
          ret[lhs_var] = rhs_var;
          visited_rhs_index.insert(i);
          break;
        }
      }
    }
    return ret;
  };

  std::unordered_map<cinn::ir::Var, cinn::ir::Var> for_var_map =
      ConstructForVarReplaceMap(src_vars, dst_vars);
  std::vector<ir::Expr> new_iter_values;
  for (const ir::Expr& iter_value : sbr->iter_values) {
    ir::Expr new_iter_value = ir::ir_utils::IRCopy(iter_value);
    for (const auto& [lhs_var, rhs_var] : for_var_map) {
      ReplaceVarWithExpr(
          &new_iter_value, lhs_var, ir::ir_utils::IRCopy(rhs_var));
    }
    new_iter_values.push_back(new_iter_value);
  }
  return ir::ScheduleBlockRealize::Make(
      new_iter_values, ir::ir_utils::IRCopy(sbr->schedule_block));
}

struct MoveScheduleBlockMutator : public ir::IRMutator<Expr*> {
 public:
  MoveScheduleBlockMutator(const std::string& src, const std::string& dst)
      : src_(src), dst_(dst) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Block* op, ir::Expr* expr) {
    auto InsertRootAndRemoveCurrentStmts = [&](ir::Block* current_block) {
      if (block_to_new_stmts_.find(current_block) !=
          block_to_new_stmts_.end()) {
        current_block->stmts = block_to_new_stmts_[current_block];
      }
      while (!insert_root_schedule_blocks_.empty()) {
        VLOG(6) << "Insert to root block: "
                << insert_root_schedule_blocks_.back();
        root_block_->stmts.insert(root_block_->stmts.begin(),
                                  insert_root_schedule_blocks_.back());
        insert_root_schedule_blocks_.pop_back();
      }
    };

    auto* node = expr->As<ir::Block>();
    current_block_ = node;
    ir::IRMutator<>::Visit(op, expr);
    InsertRootAndRemoveCurrentStmts(node);
  }

  void Visit(const ir::ScheduleBlockRealize* op, ir::Expr* expr) override {
    auto* sbr_node = expr->As<ir::ScheduleBlockRealize>();
    current_sbr_ = sbr_node;
    const auto* sb_node = sbr_node->schedule_block.As<ir::ScheduleBlock>();
    if (sb_node->name == dst_) {
      root_block_ = current_block_;
      root_for_var_extents_ = for_var_extents_;
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For* op, ir::Expr* expr) {
    auto* node = expr->As<ir::For>();
    for_var_extents_.push_back({node->loop_var, node->extent});
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::Store* op, ir::Expr* expr) {
    auto* node = expr->As<ir::Store>();
    ir::Expr sb = ir::ir_utils::IRCopy(current_sbr_->schedule_block);
    ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
    if (sb_node->name == src_) {
      MoveAfter();
    }
  }

  void MoveAfter() {
    // Merge current sbr to root block.
    insert_root_schedule_blocks_.push_back(
        ReplaceSbrIter(current_sbr_, for_var_extents_, root_for_var_extents_));

    // Record and will remove current sbr later.
    block_to_new_stmts_[current_block_] = [&]() -> std::vector<ir::Expr> {
      std::vector<ir::Expr> new_stmts;
      for (const ir::Expr& expr : current_block_->stmts) {
        if (expr.As<ir::ScheduleBlockRealize>()) {
          const ir::Expr sb = ir::ir_utils::IRCopy(
              expr.As<ir::ScheduleBlockRealize>()->schedule_block);
          const ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
          if (sb_node->name == src_) {
            continue;
          }
        }
        new_stmts.push_back(expr);
      }
      return new_stmts;
    }();
  }

  std::vector<ForVarExtent> for_var_extents_;
  std::vector<ForVarExtent> root_for_var_extents_;
  std::vector<ir::Expr> insert_root_schedule_blocks_;
  std::unordered_map<ir::Block*, std::vector<ir::Expr>> block_to_new_stmts_;

  ir::Block* root_block_;
  ir::Block* current_block_;
  ir::ScheduleBlockRealize* current_sbr_;
  const std::string& src_;
  const std::string& dst_;
};

struct EmptyBlockRemover : public ir::IRMutator<Expr*> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Block* op, ir::Expr* expr) {
    auto* node = expr->As<ir::Block>();
    std::unordered_set<int> need_remove_ids;
    for (int i = 0; i < node->stmts.size(); ++i) {
      if (IsEmptyBlock(node->stmts[i]) || IsEmptyFor(node->stmts[i])) {
        need_remove_ids.insert(i);
      }
    }
    if (!need_remove_ids.empty()) {
      node->stmts = [&] {
        std::vector<ir::Expr> new_stmts;
        for (int i = 0; i < node->stmts.size(); ++i) {
          if (need_remove_ids.count(i) == 0) {
            new_stmts.push_back(node->stmts[i]);
          }
        }
        return new_stmts;
      }();
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  bool IsEmptyBlock(const ir::Expr& expr) {
    const auto* block_node = expr.As<ir::Block>();
    if (block_node == nullptr) return false;
    for (const auto& stmt : block_node->stmts) {
      if (!IsEmptyBlock(stmt)) return false;
    }
    return true;
  }

  bool IsEmptyFor(const ir::Expr& expr) {
    const auto* for_node = expr.As<ir::For>();
    if (for_node == nullptr) return false;
    return IsEmptyBlock(for_node->body);
  }
};

ir::Expr LoopFusionHelper(const ir::Expr& src, const ir::Expr& dst) {
  std::vector<ForVarExtent> src_for_vars;
  std::vector<ForVarExtent> dst_for_vars;

  auto FuseLoop = [&](const ir::Expr& src, const ir::Expr& dst) -> ir::Expr {
    const auto* src_node = src.As<ir::For>();
    const auto* dst_node = dst.As<ir::For>();
    ir::Expr fused_loop = ir::ir_utils::IRCopy(dst);
    auto* fused_loop_node = fused_loop.As<ir::For>();

    src_for_vars.push_back({src_node->loop_var, src_node->extent});
    dst_for_vars.push_back({dst_node->loop_var, dst_node->extent});
    ir::Expr fused_body = LoopFusionHelper(src_node->body, dst_node->body);
    dst_for_vars.pop_back();
    src_for_vars.pop_back();

    if (fused_body == nullptr) return nullptr;
    fused_loop_node->body = fused_body;
    return fused_loop;
  };

  auto FuseBlock = [&](const ir::Expr& src, const ir::Expr& dst) -> ir::Expr {
    const auto* src_node = src.As<ir::Block>();
    const auto* dst_node = dst.As<ir::Block>();
    ir::Expr fused_block = ir::ir_utils::IRCopy(dst);
    auto* fused_block_node = fused_block.As<ir::Block>();
    // currently support blocks have equal stmt size.
    if (src_node->stmts.size() != dst_node->stmts.size()) {
      return nullptr;
    }
    PADDLE_ENFORCE_EQ(src_node->stmts.size(),
                      dst_node->stmts.size(),
                      ::common::errors::InvalidArgument(
                          "The stmts size in block should be equal."));

    for (size_t i = 0; i < dst_node->stmts.size(); ++i) {
      if (src_node->stmts[i].As<ir::ScheduleBlockRealize>() &&
          dst_node->stmts[i].As<ir::ScheduleBlockRealize>()) {
        fused_block_node->stmts.push_back(
            ReplaceSbrIter(src_node->stmts[i].As<ir::ScheduleBlockRealize>(),
                           src_for_vars,
                           dst_for_vars));
        continue;
      }
      ir::Expr fused_body =
          LoopFusionHelper(src_node->stmts[i], dst_node->stmts[i]);
      if (fused_body == nullptr) {
        return nullptr;
      }
      fused_block_node->stmts[i] = fused_body;
    }
    return fused_block;
  };

  if (src.As<ir::For>() && dst.As<ir::For>()) {
    return FuseLoop(src, dst);
  }
  if (src.As<ir::Block>() && dst.As<ir::Block>()) {
    return FuseBlock(src, dst);
  }
  return nullptr;
}

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

void MoveScheduleBlock(const std::string& src,
                       const std::string& dst,
                       ir::Expr* root) {
  MoveScheduleBlockMutator(src, dst)(root);
  EmptyBlockRemover()(root);
}

ir::Expr LoopFusion(const ir::Expr& src, const ir::Expr& dst) {
  VLOG(6) << "loop src: \n" << src;
  VLOG(6) << "loop dst: \n" << dst;
  ir::Expr fused_loop = LoopFusionHelper(src, dst);
  VLOG(6) << "fused loop: \n" << fused_loop;
  return fused_loop;
}

}  // namespace optim
}  // namespace cinn
