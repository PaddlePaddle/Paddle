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

#include "paddle/cinn/optim/eliminate_common_global_tensor.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

namespace {

struct ForVarExtent {
  ir::Var loop_var;
  ir::Expr extent;
};

struct IndicesAndExtent {
  std::vector<ir::Expr> indices;
  std::vector<ForVarExtent> for_var_extents;
};

struct GlobalTensorInfoCollector : public ir::IRMutator<Expr*> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  std::unordered_set<std::string> GetEliminateTensorNames() const {
    auto IndiceToExprWithForVar =
        [&](ir::Expr indice,
            const std::unordered_map<ir::Var, ir::Var>& upper_for_to_lower_for)
        -> ir::Expr {
      ir::Expr ret = ir::ir_utils::IRCopy(indice);
      for (const auto& [upper_for, lower_for] : upper_for_to_lower_for) {
        ReplaceVarWithExpr(&ret, upper_for, ir::ir_utils::IRCopy(lower_for));
      }
      return ret;
    };

    auto IndiceAndExtentEqual =
        [&](const IndicesAndExtent& indice_and_extent1,
            const IndicesAndExtent& indice_and_extent2) -> bool {
      const auto& indice1 = indice_and_extent1.indices;
      const auto& indice2 = indice_and_extent2.indices;
      if (indice1.size() != indice2.size()) return false;

      std::unordered_map<ir::Var, ir::Var> upper_for_to_lower_for = [&]() {
        std::unordered_map<ir::Var, ir::Var> ret;
        const auto& for_var_extents1 = indice_and_extent1.for_var_extents;
        const auto& for_var_extents2 = indice_and_extent2.for_var_extents;
        std::unordered_set<std::size_t> visited_lower_extend;
        for (const auto& [upper_var, upper_extend] : for_var_extents1) {
          for (std::size_t i = 0; i < for_var_extents2.size(); ++i) {
            const auto& [lower_var, lower_extend] = for_var_extents2[i];
            if (cinn::common::AutoSimplify(
                    ir::Sub::Make(upper_extend, lower_extend)) == ir::Expr(0) &&
                visited_lower_extend.count(i) == 0) {
              ret[upper_var] = lower_var;
              visited_lower_extend.insert(i);
              break;
            }
          }
        }
        return ret;
      }();

      for (size_t i = 0; i < indice1.size(); ++i) {
        ir::Expr lhs =
            IndiceToExprWithForVar(indice1.at(i), upper_for_to_lower_for);
        ir::Expr rhs =
            IndiceToExprWithForVar(indice2.at(i), upper_for_to_lower_for);
        if (cinn::common::AutoSimplify(ir::Sub::Make(lhs, rhs)) !=
            ir::Expr(0)) {
          return false;
        }
      }
      return true;
    };

    auto AllIndiceAndExtentEqual =
        [&](const std::vector<IndicesAndExtent>& indice_and_extent) -> bool {
      CHECK_GE(indice_and_extent.size(), 2);
      for (size_t i = 1; i < indice_and_extent.size(); ++i) {
        if (!IndiceAndExtentEqual(indice_and_extent[0], indice_and_extent[i]))
          return false;
      }
      return true;
    };

    auto IsGlobalTensorNeedEliminate =
        [&](const std::vector<IndicesAndExtent>& indice_and_extent) -> bool {
      if (indice_and_extent.size() <= 1) return false;
      return AllIndiceAndExtentEqual(indice_and_extent);
    };

    std::unordered_set<std::string> global_tensor;
    for (const auto& [tensor_name, indice_and_extent] :
         tensor_to_indice_and_extent_) {
      if (IsGlobalTensorNeedEliminate(indice_and_extent)) {
        global_tensor.insert(tensor_name);
      }
    }
    return global_tensor;
  }

 private:
  void Visit(const ir::ScheduleBlockRealize* op, ir::Expr* expr) override {
    auto* sbr_node = expr->As<ir::ScheduleBlockRealize>();
    CHECK(sbr_node);
    const auto& iter_values = sbr_node->iter_values;
    auto* sb_node = sbr_node->schedule_block.As<ir::ScheduleBlock>();
    const auto& iter_vars = sb_node->iter_vars;
    CHECK_EQ(iter_values.size(), iter_vars.size());
    for (std::size_t i = 0; i < iter_values.size(); ++i) {
      var_to_value_expr_[iter_vars[i]] = iter_values[i];
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::For>();
    CHECK(node);
    for_var_extents_.push_back({node->loop_var, node->extent});
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::Load* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Load>();
    CHECK(node);
    const auto& load_buffer = node->tensor.as_tensor_ref()->buffer;
    if (load_buffer->memory_type == ir::MemoryType::Heap) {
      std::vector<ir::Expr> indices;
      for (const auto& indice : node->indices) {
        ir::Expr new_indice = ir::ir_utils::IRCopy(indice);
        for (const auto& [var, value_expr] : var_to_value_expr_) {
          ReplaceVarWithExpr(
              &new_indice, var, ir::ir_utils::IRCopy(value_expr));
        }
        indices.push_back(new_indice);
      }

      tensor_to_indice_and_extent_[load_buffer->name].push_back(
          {indices, for_var_extents_});
    }
  }

  std::vector<ForVarExtent> for_var_extents_;
  std::unordered_map<ir::Var, ir::Expr> var_to_value_expr_;
  std::unordered_map<std::string, std::vector<IndicesAndExtent>>
      tensor_to_indice_and_extent_;
};

struct CommonGlobalTensorEliminator : public ir::IRMutator<Expr*> {
  CommonGlobalTensorEliminator(
      const std::unordered_set<std::string>& eliminate_tensor_names)
      : eliminate_tensor_names_(eliminate_tensor_names) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Block* op, Expr* expr) override {
    auto* node = expr->As<ir::Block>();
    CHECK(node);
    current_block_ = node;
    IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override {
    auto* node = expr->As<ir::ScheduleBlockRealize>();
    CHECK(node);
    current_sbr_ = node;
    IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load* op, Expr* expr) override {
    auto* node = expr->As<ir::Load>();
    CHECK(node);
    const auto& buffer_name = node->tensor.as_tensor_ref()->buffer->name;
    if (eliminate_tensor_names_.count(buffer_name) == 0) {
      return;
    }

    if (global_tensor_to_local_tensor_.count(buffer_name) == 0) {
      InsertLocalTensorBlock(node, buffer_name);
    }
    SubstituteGlobalTensor(node, buffer_name);
  }

  void InsertLocalTensorBlock(ir::Load* load_node,
                              const std::string& buffer_name) {
    ir::Expr sb = ir::ir_utils::IRCopy(current_sbr_->schedule_block);
    ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
    CHECK(sb_node);

    const auto& old_tensor = load_node->tensor.as_tensor_ref();
    ir::Expr new_tensor =
        ir::_Tensor_::Make(old_tensor->name + "_local",
                           old_tensor->type(),
                           ir::ir_utils::IRCopy(old_tensor->shape),
                           ir::ir_utils::IRCopy(old_tensor->domain),
                           old_tensor->reduce_axis);
    new_tensor.as_tensor_ref()->WithBuffer(
        "local", new_tensor.as_tensor_ref()->name + "_buffer");
    ir::Expr new_body =
        ir::Store::Make(new_tensor,
                        ir::ir_utils::IRCopy(ir::Expr(load_node)),
                        ir::ir_utils::IRCopy(load_node->indices));
    ir::Expr new_sb = ir::ScheduleBlock::Make(
        sb_node->iter_vars, {}, {}, sb_node->name + "_local", new_body);

    ir::Expr new_sbr = ir::ScheduleBlockRealize::Make(
        ir::ir_utils::IRCopy(current_sbr_->iter_values), new_sb);
    CHECK_EQ(global_tensor_to_local_tensor_.count(buffer_name), 0);
    global_tensor_to_local_tensor_[buffer_name] = new_tensor;
    current_block_->stmts.insert(current_block_->stmts.begin(), new_sbr);
  }

  void SubstituteGlobalTensor(ir::Load* load_node,
                              const std::string& buffer_name) {
    CHECK_GT(global_tensor_to_local_tensor_.count(buffer_name), 0);
    load_node->tensor = global_tensor_to_local_tensor_[buffer_name];
  }

 private:
  std::unordered_set<std::string> eliminate_tensor_names_;
  std::unordered_map<std::string, ir::Expr> global_tensor_to_local_tensor_;

  ir::Block* current_block_;
  ir::ScheduleBlockRealize* current_sbr_;
};

}  // namespace

void EliminateCommonGlobalTensor(Expr* e) {
  GlobalTensorInfoCollector collector;
  collector(e);

  const auto& eliminate_tensor_names = collector.GetEliminateTensorNames();

  CommonGlobalTensorEliminator eliminator(eliminate_tensor_names);
  eliminator(e);
}

}  // namespace optim
}  // namespace cinn
