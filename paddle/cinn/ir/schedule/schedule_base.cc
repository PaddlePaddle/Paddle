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

bool IsContigous(const std::vector<int>& merge_index,
                 const std::vector<Expr>& stride,
                 const std::vector<Expr>& view_shape) {
  for (size_t i = 0; i + 1 < merge_index.size(); ++i) {
    if (cinn::common::AutoSimplify(stride[i]) !=
        cinn::common::AutoSimplify(stride[i + 1] * view_shape[i + 1])) {
      return false;
    }
  }

  return true;
}

void ScheduleBase::UpdateMergeOffset(const std::vector<int>& merge_index,
                                     const Expr& new_indice) {
  struct LoadStoreOffsetMutator : public ir::IRMutator<> {
    LoadStoreOffsetMutator(const std::vector<int> merge_index,
                           const Expr& new_indice)
        : merge_index_(merge_index), new_indice_(new_indice) {}

    void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

   private:
    void Visit(const ir::Store* op, Expr* expr) override {
      auto* node = expr->As<ir::Store>();
      auto* tensor = node->tensor.as_tensor();

      auto& view_shape = node->view_shape;

      if (!IsContigous(merge_index_, node->stride_info, node->view_shape)) {
        PADDLE_THROW(::common::errors::Unimplemented(
            "Only support contigous merge yet"));
      }

      std::set<int> merge_set(merge_index_.begin(), merge_index_.end());

      Expr new_dim = Expr(1);
      for (int i = 0; i < view_shape.size(); ++i) {
        if (merge_set.count(i)) {
          new_dim = new_dim * view_shape[i];
        }
      }
      new_dim = cinn::common::AutoSimplify(new_dim);

      // update loop_var, stride_info, view_shape
      Expr update_stride = node->stride_info[merge_index_.back()];

      bool insert_flags = false;
      std::vector<Expr> new_shape;
      std::vector<Expr> new_loop_var;
      std::vector<Expr> new_stride_info;
      for (int i = 0; i < view_shape.size(); ++i) {
        if (merge_set.count(i)) {
          if (!insert_flags) {
            new_shape.push_back(new_dim);
            new_loop_var.push_back(new_indice_);
            new_stride_info.push_back(update_stride);
          }
          insert_flags = true;
        } else {
          new_shape.push_back(view_shape[i]);
          new_loop_var.push_back(node->loop_vars[i]);
          new_stride_info.push_back(node->stride_info[i]);
        }
      }

      auto new_offset =
          cinn::common::IndiceToAbsOffset(new_shape, new_loop_var);

      std::cerr << "new offset \t" << new_offset << std::endl;
      // cinn::ir::ReplaceExpr( &(node->offset), {base_expr_}, {new_offset_});
      //  node->offset = new_offset_;
      std::cerr << "update offset !!! " << node->offset << std::endl;

      node->view_shape = new_shape;
      node->loop_vars = new_loop_var;
      node->stride_info = new_stride_info;
      node->offset = new_offset;
    }

   private:
    std::vector<int> merge_index_;
    Expr new_indice_;
  };

  LoadStoreOffsetMutator load_store_mutator(merge_index, new_indice);

  auto exprs = module_expr_.GetExprs();

  for (auto& i : exprs) {
    load_store_mutator(&i);
  }
}

void ScheduleBase::UpdateSplitOffset(const Var& base_loop_var,
                                     const std::vector<Var>& new_looop_vars,
                                     const std::vector<int>& split_factors) {
  struct LoadStoreOffsetMutator : public ir::IRMutator<> {
    LoadStoreOffsetMutator(const Var& base_loop_var,
                           const std::vector<Var>& new_looop_vars,
                           const std::vector<int>& split_factors)
        : base_loop_var_(base_loop_var),
          new_loop_vars_(new_looop_vars),
          split_factors_(split_factors) {}

    void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

   private:
    void Visit(const ir::Store* op, Expr* expr) override {
      auto* node = expr->As<ir::Store>();
      auto* tensor = node->tensor.as_tensor();

      std::cerr << "before replace " << node->offset << std::endl;
      Expr update_loop_var = Expr(0);

      std::cerr << "base loop var name " << base_loop_var_->name << std::endl;

      int split_index = -1;
      for (int i = 0; i < node->loop_vars.size(); ++i) {
        if (node->loop_vars[i].As<ir::_Var_>()->name == base_loop_var_->name) {
          split_index = i;
          break;
        }
      }

      PADDLE_ENFORCE_NE(
          split_index,
          -1,
          ::common::errors::PreconditionNotMet("split index can not be -1"));

      std::vector<Expr> new_shape;
      std::vector<Expr> new_loop_var;
      std::vector<Expr> new_stride_info;

      std::vector<Expr> split_stride;
      Expr base = node->stride_info[split_index];

      for (int i = split_factors_.size() - 1; i >= 0; --i) {
        split_stride.insert(split_stride.begin(), base);

        base = base * Expr(split_factors_[i]);
      }

      for (int i = 0; i < node->view_shape.size(); ++i) {
        if (i == split_index) {
          for (size_t j = 0; j < new_loop_vars_.size(); ++j) {
            new_shape.push_back(Expr(split_factors_[j]));
            new_loop_var.push_back(new_loop_vars_[j]);

            new_stride_info.push_back(split_stride[j]);
          }
        } else {
          new_shape.push_back(node->view_shape[i]);
          new_loop_var.push_back(node->loop_vars[i]);
          new_stride_info.push_back(node->stride_info[i]);
        }
      }

      auto new_offset =
          cinn::common::IndiceToAbsOffset(new_shape, new_loop_var);

      std::cerr << "new cal offset " << new_offset << std::endl;

      node->offset = new_offset;

      node->view_shape = new_shape;
      node->loop_vars = new_loop_var;
      node->stride_info = new_stride_info;
      std::cerr << "after replace " << node->offset << std::endl;

      std::cerr << "node " << node->index() << std::endl;
    }

   private:
    Var base_loop_var_;
    std::vector<Var> new_loop_vars_;
    std::vector<int> split_factors_;
  };

  LoadStoreOffsetMutator load_store_mutator(
      base_loop_var, new_looop_vars, split_factors);

  auto exprs = module_expr_.GetExprs();

  for (auto& i : exprs) {
    load_store_mutator(&i);
  }
}

void ScheduleBase::UpdateReorderOffset(
    const std::vector<Var>& reorder_var_list) {
  struct LoadStoreReorderMutator : public ir::IRMutator<> {
    explicit LoadStoreReorderMutator(const std::vector<Var>& reorder_var_list)
        : reorder_var_list_(reorder_var_list) {}

    void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

   private:
    void Visit(const ir::Store* op, Expr* expr) override {
      auto* node = expr->As<ir::Store>();
      auto* tensor = node->tensor.as_tensor();

      std::map<std::string, int> reorder_map;

      std::set<std::string> var_set;
      for (size_t i = 0; i < reorder_var_list_.size(); ++i) {
        var_set.insert(reorder_var_list_[i]->name);
      }

      int split_index = -1;
      std::vector<int> pos_idx;
      for (int i = 0; i < node->loop_vars.size(); ++i) {
        if (var_set.count(node->loop_vars[i].As<ir::_Var_>()->name)) {
          reorder_map[node->loop_vars[i].As<ir::_Var_>()->name] = i;
          pos_idx.push_back(i);
        }
      }

      PADDLE_ENFORCE_EQ(reorder_map.size(),
                        reorder_var_list_.size(),
                        ::common::errors::PreconditionNotMet(
                            "all reorder var MUST in loop vars"));

      std::vector<Expr> new_shape = node->view_shape;
      std::vector<Expr> new_loop_var = node->loop_vars;
      std::vector<Expr> new_stride_info = node->stride_info;

      for (size_t i = 0; i < reorder_var_list_.size(); ++i) {
        int val_idx = reorder_map.at(reorder_var_list_[i]->name);
        new_loop_var[pos_idx[i]] = node->loop_vars[val_idx];
        new_shape[pos_idx[i]] = node->view_shape[val_idx];
        new_stride_info[pos_idx[i]] = node->stride_info[val_idx];
      }

      auto new_offset =
          cinn::common::IndiceToAbsOffset(new_shape, new_loop_var);

      std::cerr << "new cal offset " << new_offset << std::endl;

      node->offset = new_offset;

      node->view_shape = new_shape;
      node->loop_vars = new_loop_var;
      node->stride_info = new_stride_info;
      std::cerr << "after replace " << node->offset << std::endl;

      std::cerr << "node " << node->index() << std::endl;
    }

   private:
    std::vector<Var> reorder_var_list_;
  };

  LoadStoreReorderMutator load_store_reorder_mutator(reorder_var_list);

  auto exprs = module_expr_.GetExprs();

  for (auto& i : exprs) {
    load_store_reorder_mutator(&i);
  }
}

}  // namespace ir
}  // namespace cinn
