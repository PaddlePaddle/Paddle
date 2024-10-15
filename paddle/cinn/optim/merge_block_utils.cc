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

// Replace bind values in ScheduleBlockRealize using `for_var_map`.
ir::Expr ReplaceSbrIter(
    const ir::ScheduleBlockRealize* sbr,
    const std::unordered_map<ir::Var, ir::Var>& for_var_map) {
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

// Need to check dependency when dependency analysis tools are complete.
struct MoveScheduleBlockMutator : public ir::IRMutator<Expr*> {
 public:
  MoveScheduleBlockMutator(const ir::ScheduleBlock* src,
                           const ir::ScheduleBlock* dst)
      : src_(src), dst_(dst) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Block* op, ir::Expr* expr) {
    auto InsertRootAndRemoveCurrentStmts = [&](ir::Block* current_block) {
      if (block_to_new_stmts_.find(current_block) !=
          block_to_new_stmts_.end()) {
        current_block->stmts = block_to_new_stmts_[current_block];
      }
      if (insert_root_schedule_block_ != nullptr) {
        root_block_->stmts = [&]() -> std::vector<ir::Expr> {
          std::vector<ir::Expr> new_stmts;
          for (const ir::Expr& expr : root_block_->stmts) {
            new_stmts.push_back(expr);
            if (expr.As<ir::ScheduleBlockRealize>()) {
              auto* sbr = expr.As<ir::ScheduleBlockRealize>();
              auto* sb = sbr->schedule_block.As<ir::ScheduleBlock>();
              if (sb->name == dst_->name) {
                VLOG(6) << "Insert to root block: "
                        << insert_root_schedule_block_;
                new_stmts.push_back(insert_root_schedule_block_);
                insert_root_schedule_block_ = nullptr;
              }
            }
          }
          return new_stmts;
        }();
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
    if (sb_node->name == dst_->name) {
      root_block_ = current_block_;
      root_for_vars_ = for_vars_;
    }
    if (sb_node->name == src_->name) {
      MoveAfter();
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For* op, ir::Expr* expr) {
    auto* node = expr->As<ir::For>();
    for_vars_.push_back(node->loop_var);
    ir::IRMutator<>::Visit(op, expr);
    for_vars_.pop_back();
  }

  void MoveAfter() {
    auto ConstructForVarReplaceMap = [&](const std::vector<ir::Var>& lhs_vars,
                                         const std::vector<ir::Var>& rhs_vars)
        -> std::unordered_map<ir::Var, ir::Var> {
      PADDLE_ENFORCE_EQ(lhs_vars.size(),
                        rhs_vars.size(),
                        ::common::errors::InvalidArgument(
                            "The for vars size should be equal."));
      std::unordered_map<ir::Var, ir::Var> ret;
      for (std::size_t i = 0; i < lhs_vars.size(); ++i) {
        const auto& rhs_var = rhs_vars[i];
        ret[lhs_vars[i]] = rhs_var;
      }
      return ret;
    };

    // Merge current sbr to root block.
    insert_root_schedule_block_ = ReplaceSbrIter(
        current_sbr_, ConstructForVarReplaceMap(for_vars_, root_for_vars_));

    // Record and will remove current sbr later.
    block_to_new_stmts_[current_block_] = [&]() -> std::vector<ir::Expr> {
      std::vector<ir::Expr> new_stmts;
      for (const ir::Expr& expr : current_block_->stmts) {
        if (expr.As<ir::ScheduleBlockRealize>()) {
          const ir::Expr sb = ir::ir_utils::IRCopy(
              expr.As<ir::ScheduleBlockRealize>()->schedule_block);
          const ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
          if (sb_node->name == src_->name) {
            continue;
          }
        }
        new_stmts.push_back(expr);
      }
      return new_stmts;
    }();
  }

  std::vector<ir::Var> for_vars_;
  std::vector<ir::Var> root_for_vars_;
  std::unordered_map<ir::Block*, std::vector<ir::Expr>> block_to_new_stmts_;

  ir::Block* root_block_{nullptr};
  ir::Block* current_block_{nullptr};
  ir::ScheduleBlockRealize* current_sbr_{nullptr};
  ir::Expr insert_root_schedule_block_{nullptr};
  const ir::ScheduleBlock* src_;
  const ir::ScheduleBlock* dst_;
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

struct LoopFusionFunctor {
 public:
  ir::Expr FuseTwoLoops(const ir::For* src, const ir::For* dst) {
    return LoopFusionImpl(src, dst);
  }

 private:
  ir::Expr LoopFusionImpl(const ir::Expr& src, const ir::Expr& dst) {
    // Need to check dependency when dependency analysis tools are complete.
    if (src.As<ir::For>() && dst.As<ir::For>()) {
      return LoopFusionImpl(src.As<ir::For>(), dst.As<ir::For>());
    }
    if (src.As<ir::Block>() && dst.As<ir::Block>()) {
      return LoopFusionImpl(src.As<ir::Block>(), dst.As<ir::Block>());
    }
    return nullptr;
  }

  ir::Expr LoopFusionImpl(const ir::For* src, const ir::For* dst) {
    ir::Expr fused_loop = ir::For::Make(dst->loop_var,
                                        dst->min,
                                        dst->extent,
                                        dst->for_type(),
                                        dst->device_api,
                                        dst->body,
                                        dst->vectorize_info(),
                                        dst->bind_info());
    auto* fused_loop_node = fused_loop.As<ir::For>();

    src_to_dst_for_vars_[src->loop_var] = dst->loop_var;
    ir::Expr fused_body = LoopFusionImpl(src->body, dst->body);

    if (fused_body == nullptr) return nullptr;
    fused_loop_node->body = fused_body;
    return fused_loop;
  }

  ir::Expr LoopFusionImpl(const ir::Block* src, const ir::Block* dst) {
    ir::Expr fused_block = ir::Block::Make(dst->stmts);
    auto* fused_block_node = fused_block.As<ir::Block>();
    // Currently support blocks have equal stmts size.
    if (src->stmts.size() != dst->stmts.size()) {
      return nullptr;
    }
    PADDLE_ENFORCE_EQ(src->stmts.size(),
                      dst->stmts.size(),
                      ::common::errors::InvalidArgument(
                          "The stmts size in block should be equal."));

    for (size_t i = 0; i < dst->stmts.size(); ++i) {
      if (src->stmts[i].As<ir::ScheduleBlockRealize>() &&
          dst->stmts[i].As<ir::ScheduleBlockRealize>()) {
        fused_block_node->stmts.push_back(
            ReplaceSbrIter(src->stmts[i].As<ir::ScheduleBlockRealize>(),
                           src_to_dst_for_vars_));
        continue;
      }
      ir::Expr fused_body = LoopFusionImpl(src->stmts[i], dst->stmts[i]);
      if (fused_body == nullptr) {
        return nullptr;
      }
      fused_block_node->stmts[i] = fused_body;
    }
    return fused_block;
  }

  std::unordered_map<ir::Var, ir::Var> src_to_dst_for_vars_;
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

void MoveScheduleBlock(const ir::ScheduleBlock* src,
                       const ir::ScheduleBlock* dst,
                       ir::Expr* root) {
  MoveScheduleBlockMutator(src, dst)(root);
  EmptyBlockRemover()(root);
}

ir::Expr LoopFusion(const ir::For* src, const ir::For* dst) {
  VLOG(6) << "Begin LoopFusion: \n";
  ir::Expr fused_loop = LoopFusionFunctor().FuseTwoLoops(src, dst);
  if (fused_loop != nullptr) {
    VLOG(6) << "After LoopFusion, Fused loop: \n" << fused_loop;
  } else {
    VLOG(6) << "Not supported for those loops!";
  }
  return fused_loop;
}

}  // namespace optim
}  // namespace cinn
