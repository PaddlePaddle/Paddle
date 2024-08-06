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

void ScheduleBase::UpdateMergeOffset(const std::string& block_name,
                                     const std::vector<int>& merge_index,
                                     const Expr& new_indice) {
  struct LoadStoreOffsetMutator : public ir::IRMutator<> {
    LoadStoreOffsetMutator(const std::string& block_name,
                           const std::vector<int> merge_index,
                           const Expr& new_indice)
        : block_name_(block_name),
          merge_index_(merge_index),
          new_indice_(new_indice) {}

    void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

   private:
    void UpdateInnerInfo(std::vector<Expr>* loop_vars,
                         std::vector<Expr>* view_shape,
                         std::vector<Expr>* stride_info) {
      std::cerr << "loop vars " << std::endl;
      for (auto val : *loop_vars) {
        std::cerr << "var " << val << std::endl;
      }

      std::cerr << "shape info " << std::endl;
      for (auto val : *view_shape) {
        std::cerr << "shape " << val << std::endl;
      }

      std::cerr << "stride info " << std::endl;
      for (auto val : *stride_info) {
        std::cerr << "stride " << val << std::endl;
      }

      std::cerr << "merge index \n";
      for (auto val : merge_index_) {
        std::cerr << "merge index " << val << std::endl;
      }

      if (!IsContigous(merge_index_, *stride_info, *view_shape)) {
        PADDLE_THROW(::common::errors::Unimplemented(
            "Only support contigous merge yet"));
      }

      std::set<int> merge_set(merge_index_.begin(), merge_index_.end());

      Expr new_dim = Expr(1);
      for (int i = 0; i < view_shape->size(); ++i) {
        if (merge_set.count(i)) {
          new_dim = new_dim * (*view_shape)[i];
        }
      }
      new_dim = cinn::common::AutoSimplify(new_dim);

      // update loop_var, stride_info, view_shape
      Expr update_stride = (*stride_info)[merge_index_.back()];

      bool insert_flags = false;
      std::vector<Expr> new_shape;
      std::vector<Expr> new_loop_var;
      std::vector<Expr> new_stride_info;
      for (int i = 0; i < view_shape->size(); ++i) {
        if (merge_set.count(i)) {
          if (!insert_flags) {
            new_shape.push_back(new_dim);
            new_loop_var.push_back(new_indice_);
            new_stride_info.push_back(update_stride);
          }
          insert_flags = true;
        } else {
          new_shape.push_back((*view_shape)[i]);
          new_loop_var.push_back((*loop_vars)[i]);
          new_stride_info.push_back((*stride_info)[i]);
        }
      }

      view_shape->swap(new_shape);
      loop_vars->swap(new_loop_var);
      stride_info->swap(new_stride_info);

      std::cerr << "after merge loop vars !!! " << std::endl;
      for (auto& var : *loop_vars) {
        std::cerr << "loop var " << var << std::endl;
      }
    }

    void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) {
      auto block_name = expr->As<ir::ScheduleBlockRealize>()
                            ->schedule_block.As<ScheduleBlock>()
                            ->name;
      if ((block_name.substr(0, 4) != "root") && (block_name != block_name_)) {
        std::cerr << "skip here !!!!!!! "
                  << expr->As<ir::ScheduleBlockRealize>()
                         ->schedule_block.As<ScheduleBlock>()
                         ->name
                  << std::endl;
        return;
      }

      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::Store* op, Expr* expr) override {
      auto* node = expr->As<ir::Store>();
      auto* tensor = node->tensor.as_tensor();

      std::cerr << "store tensor info " << tensor->name << std::endl;
      UpdateInnerInfo(
          &(node->loop_vars), &(node->view_shape), &(node->stride_info));

      std::cerr << "before merge info !! " << node->offset() << std::endl;

      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::Load* op, Expr* expr) override {
      auto* node = expr->As<ir::Load>();
      auto* tensor = node->tensor.as_tensor();

      std::cerr << "load tensor info " << tensor->name << std::endl;

      std::cerr << "before merge load info !! " << node->offset() << std::endl;
      UpdateInnerInfo(
          &(node->loop_vars), &(node->view_shape), &(node->stride_info));

      std::cerr << "after merge load info !! " << node->offset() << std::endl;

      ir::IRMutator<>::Visit(op, expr);
    }

   private:
    std::string block_name_;
    std::vector<int> merge_index_;
    Expr new_indice_;
  };

  LoadStoreOffsetMutator load_store_mutator(
      block_name, merge_index, new_indice);

  auto exprs = module_expr_.GetExprs();

  for (auto& i : exprs) {
    load_store_mutator(&i);
  }
}

void ScheduleBase::UpdateSplitOffset(const std::string& block_id,
                                     const Var& base_loop_var,
                                     const std::vector<Var>& new_looop_vars,
                                     const std::vector<int>& split_factors) {
  struct LoadStoreOffsetMutator : public ir::IRMutator<> {
    LoadStoreOffsetMutator(const std::string& block_id,
                           const Var& base_loop_var,
                           const std::vector<Var>& new_looop_vars,
                           const std::vector<int>& split_factors)
        : block_id_(block_id),
          base_loop_var_(base_loop_var),
          new_loop_vars_(new_looop_vars),
          split_factors_(split_factors) {}

    void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

   private:
    void UpdateInnerInfo(std::vector<Expr>* loop_vars,
                         std::vector<Expr>* view_shape,
                         std::vector<Expr>* stride_info) {
      Expr update_loop_var = Expr(0);

      std::cerr << "base loop var name " << base_loop_var_->name << std::endl;

      int split_index = -1;
      for (int i = 0; i < loop_vars->size(); ++i) {
        std::cerr << "loop var " << i << "\t" << loop_vars[i] << std::endl;
        if ((*loop_vars)[i].As<ir::_Var_>()->name == base_loop_var_->name) {
          split_index = i;
          break;
        }
      }

      if (split_index == -1) {
        return;
      }
      std::cerr << "split index " << split_index << std::endl;

      std::vector<Expr> new_shape;
      std::vector<Expr> new_loop_var;
      std::vector<Expr> new_stride_info;

      std::vector<Expr> split_stride;
      Expr base = (*stride_info)[split_index];

      for (int i = split_factors_.size() - 1; i >= 0; --i) {
        split_stride.insert(split_stride.begin(), base);

        base = base * Expr(split_factors_[i]);
      }

      std::cerr << "split index 11" << std::endl;
      for (int i = 0; i < view_shape->size(); ++i) {
        if (i == split_index) {
          for (size_t j = 0; j < new_loop_vars_.size(); ++j) {
            new_shape.push_back(Expr(split_factors_[j]));
            new_loop_var.push_back(new_loop_vars_[j]);

            new_stride_info.push_back(split_stride[j]);
          }
        } else {
          new_shape.push_back((*view_shape)[i]);
          new_loop_var.push_back((*loop_vars)[i]);
          new_stride_info.push_back((*stride_info)[i]);
        }
      }

      view_shape->swap(new_shape);
      loop_vars->swap(new_loop_var);
      stride_info->swap(new_stride_info);
    }

    void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) {
      auto block_name = expr->As<ir::ScheduleBlockRealize>()
                            ->schedule_block.As<ScheduleBlock>()
                            ->name;
      if ((block_name.substr(0, 4) != "root") && (block_name != block_id_)) {
        std::cerr << "skip here !!!!!!! "
                  << expr->As<ir::ScheduleBlockRealize>()
                         ->schedule_block.As<ScheduleBlock>()
                         ->name
                  << std::endl;
        return;
      }

      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::Store* op, Expr* expr) override {
      auto* node = expr->As<ir::Store>();
      auto* tensor = node->tensor.as_tensor();

      std::cerr << "before  store split schedule !!!!!!!!  " << node->offset()
                << "\t" << node->tensor << std::endl;
      UpdateInnerInfo(
          &(node->loop_vars), &(node->view_shape), &(node->stride_info));

      std::cerr << "after store split schedule !!!!!!!!  " << node->offset()
                << std::endl;

      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::Load* op, Expr* expr) override {
      auto* node = expr->As<ir::Load>();
      auto* tensor = node->tensor.as_tensor();
      std::cerr << "before load split schedule !!!!!!!!  " << node->offset()
                << "\t" << node->tensor << std::endl;
      UpdateInnerInfo(
          &(node->loop_vars), &(node->view_shape), &(node->stride_info));

      std::cerr << "after load split schedule !!!!!!!!" << node->offset()
                << std::endl;

      ir::IRMutator<>::Visit(op, expr);
    }

   private:
    std::string block_id_;
    Var base_loop_var_;
    std::vector<Var> new_loop_vars_;
    std::vector<int> split_factors_;
  };

  LoadStoreOffsetMutator load_store_mutator(
      block_id, base_loop_var, new_looop_vars, split_factors);

  auto exprs = module_expr_.GetExprs();

  for (auto& i : exprs) {
    load_store_mutator(&i);
  }
}

void ScheduleBase::UpdateReorderOffset(
    const std::string& block_name, const std::vector<Var>& reorder_var_list) {
  struct LoadStoreReorderMutator : public ir::IRMutator<> {
    explicit LoadStoreReorderMutator(const std::string& block_name,
                                     const std::vector<Var>& reorder_var_list)
        : block_name_(block_name), reorder_var_list_(reorder_var_list) {}

    void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

   private:
    void UpdateInnerInfo(std::vector<Expr>* loop_vars,
                         std::vector<Expr>* view_shape,
                         std::vector<Expr>* stride_info) {
      std::map<std::string, int> reorder_map;

      std::set<std::string> var_set;
      for (size_t i = 0; i < reorder_var_list_.size(); ++i) {
        std::cerr << "reorder var list name " << reorder_var_list_[i]->name
                  << std::endl;
        var_set.insert(reorder_var_list_[i]->name);
      }

      int split_index = -1;
      std::vector<int> pos_idx;
      for (int i = 0; i < loop_vars->size(); ++i) {
        std::cerr << " !!!!!!!! loop vars " << (*loop_vars)[i] << std::endl;
        //<< (*loop_vars)[i].As<ir::_Var_>()->name << std::endl;
        if (var_set.count((*loop_vars)[i].As<ir::_Var_>()->name)) {
          reorder_map[(*loop_vars)[i].As<ir::_Var_>()->name] = i;
          pos_idx.push_back(i);
        }
      }
      if (reorder_map.size() == 0) {
        return;
      }
      PADDLE_ENFORCE_EQ(reorder_map.size(),
                        reorder_var_list_.size(),
                        ::common::errors::PreconditionNotMet(
                            "all reorder var MUST in loop vars"));

      std::vector<Expr> new_shape = *view_shape;
      std::vector<Expr> new_loop_var = *loop_vars;
      std::vector<Expr> new_stride_info = *stride_info;

      for (size_t i = 0; i < reorder_var_list_.size(); ++i) {
        int val_idx = reorder_map.at(reorder_var_list_[i]->name);
        new_loop_var[pos_idx[i]] = (*loop_vars)[val_idx];
        new_shape[pos_idx[i]] = (*view_shape)[val_idx];
        new_stride_info[pos_idx[i]] = (*stride_info)[val_idx];
      }

      auto new_offset =
          cinn::common::IndiceToAbsOffset(new_shape, new_loop_var);

      std::cerr << "new cal offset " << new_offset << std::endl;

      view_shape->swap(new_shape);
      loop_vars->swap(new_loop_var);
      stride_info->swap(new_stride_info);
    }

    void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) {
      auto block_name = expr->As<ir::ScheduleBlockRealize>()
                            ->schedule_block.As<ScheduleBlock>()
                            ->name;
      if ((block_name.substr(0, 4) != "root") && (block_name != block_name_)) {
        std::cerr << "skip here !!!!!!! "
                  << expr->As<ir::ScheduleBlockRealize>()
                         ->schedule_block.As<ScheduleBlock>()
                         ->name
                  << std::endl;
        return;
      }

      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::Store* op, Expr* expr) override {
      auto* node = expr->As<ir::Store>();
      auto* tensor = node->tensor.as_tensor();

      UpdateInnerInfo(
          &(node->loop_vars), &(node->view_shape), &(node->stride_info));

      std::cerr << "after replace " << node->offset() << std::endl;

      ir::IRMutator<>::Visit(op, expr);

      std::cerr << "node " << node->index() << std::endl;
    }

    void Visit(const ir::Load* op, Expr* expr) override {
      auto* node = expr->As<ir::Load>();
      auto* tensor = node->tensor.as_tensor();

      UpdateInnerInfo(
          &(node->loop_vars), &(node->view_shape), &(node->stride_info));

      std::cerr << "after replace " << node->offset() << std::endl;

      std::cerr << "node " << node->index() << std::endl;

      ir::IRMutator<>::Visit(op, expr);
    }

   private:
    std::string block_name_;
    std::vector<Var> reorder_var_list_;
  };

  LoadStoreReorderMutator load_store_reorder_mutator(block_name,
                                                     reorder_var_list);

  auto exprs = module_expr_.GetExprs();

  for (auto& i : exprs) {
    load_store_reorder_mutator(&i);
  }
}

}  // namespace ir
}  // namespace cinn
