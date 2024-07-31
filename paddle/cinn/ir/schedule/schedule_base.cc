// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/schedule/schedule_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"

namespace cinn {
namespace ir {

/**
 * Replace a node to another node.
 * @param src_sref The node to be changed.
 * @param tgt_stmt The node we want.
 */
void ScheduleBase::Replace(const Expr& src_sref, const Expr& tgt_stmt) {
  CHECK(src_sref.As<ir::For>() || src_sref.As<ir::Block>() ||
        src_sref.As<ir::ScheduleBlockRealize>());
  CHECK(tgt_stmt.As<ir::For>() || tgt_stmt.As<ir::Block>() ||
        tgt_stmt.As<ir::ScheduleBlockRealize>());
  if (src_sref == tgt_stmt) {
    return;
  }
  struct ForLoopMutator : public ir::IRMutator<> {
    ForLoopMutator(const Expr& source, const Expr& target)
        : source_(source), target_(target) {}

    void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

    void Visit(const ir::For* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::Block* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }

    const Expr& source_;
    const Expr& target_;
  };
  auto exprs = module_expr_.GetExprs();
  ForLoopMutator mutator(src_sref, tgt_stmt);
  for (auto& i : exprs) {
    mutator(&i);
  }
}

void ScheduleBase::UpdateMergeOffset(const std::vector<int> merge_index,
                                     const std::vector<Expr>& new_indices) {
  struct LoadStoreOffsetMutator : public ir::IRMutator<> {
    LoadStoreOffsetMutator(const std::vector<int> merge_index,
                           const std::vector<Expr>& new_indices)
        : merge_index_(merge_index), new_indices_(new_indices) {}

    void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

   private:
    void Visit(const ir::Store* op, Expr* expr) override {
      auto* node = expr->As<ir::Store>();
      auto* tensor = node->tensor.as_tensor();

      auto& view_shape = node->view_shape;
      for (auto& s : view_shape) {
        std::cerr << "ssssssss ! " << s << std::endl;
      }

      std::set<int> merge_set(merge_index_.begin(), merge_index_.end());

      Expr base = Expr(1);
      for (int i = 0; i < view_shape.size(); ++i) {
        if (merge_set.count(i)) {
          base = base * view_shape[i];
        }
      }
      base = cinn::common::AutoSimplify(base);

      std::cerr << "merge shape " << base << std::endl;

      bool insert_flags = false;
      std::vector<Expr> new_shape;
      for (int i = 0; i < view_shape.size(); ++i) {
        if (merge_set.count(i)) {
          if (!insert_flags) {
            new_shape.push_back(base);
          }
          insert_flags = true;
        } else {
          new_shape.push_back(view_shape[i]);
        }
      }

      for (auto& s : new_shape) {
        std::cerr << "sss " << s << std::endl;
      }

      for (auto& i : new_indices_) {
        std::cerr << "iii " << i << std::endl;
      }

      auto new_offset =
          cinn::common::IndiceToAbsOffset(new_shape, new_indices_);

      std::cerr << "new offset \t" << new_offset << std::endl;
      // cinn::ir::ReplaceExpr( &(node->offset), {base_expr_}, {new_offset_});
      //  node->offset = new_offset_;
      std::cerr << "update offset !!! " << node->offset << std::endl;

      node->view_shape = new_shape;
      node->offset = new_offset;
    }

   private:
    std::vector<int> merge_index_;
    std::vector<Expr> new_indices_;
  };

  LoadStoreOffsetMutator load_store_mutator(merge_index, new_indices);

  auto exprs = module_expr_.GetExprs();

  for (auto& i : exprs) {
    load_store_mutator(&i);
  }
}

void ScheduleBase::UpdateSplitOffset(const Var& base_loop_var,
                                     const Expr& update_loop_var) {
  struct LoadStoreOffsetMutator : public ir::IRMutator<> {
    LoadStoreOffsetMutator(const Var& base_loop_var,
                           const Expr& update_loop_var)
        : base_loop_var_(base_loop_var), update_loop_var_(update_loop_var) {}

    void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

   private:
    void Visit(const ir::Store* op, Expr* expr) override {
      auto* node = expr->As<ir::Store>();
      auto* tensor = node->tensor.as_tensor();

      std::cerr << "before replace " << node->offset << std::endl;
      ReplaceExpr(&(node->offset), {base_loop_var_}, {update_loop_var_});

      std::cerr << "after replace " << node->offset << std::endl;

      std::cerr << "node " << node->index() << std::endl;
    }

   private:
    Var base_loop_var_;
    Expr update_loop_var_;
  };

  LoadStoreOffsetMutator load_store_mutator(base_loop_var, update_loop_var);

  auto exprs = module_expr_.GetExprs();

  std::cerr << "base loop var and new update loop var " << base_loop_var << "\t"
            << update_loop_var << std::endl;
  for (auto& i : exprs) {
    load_store_mutator(&i);
  }
}

}  // namespace ir
}  // namespace cinn
