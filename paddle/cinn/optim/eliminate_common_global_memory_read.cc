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

#include "paddle/cinn/optim/eliminate_common_global_memory_read.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/common/enforce.h"

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

struct ConditionForVar {
  ir::Expr condition;
  std::vector<ForVarExtent> for_var_extents;
};

/**
 * Store condition operator or if expr (e.g. condition ? true_branch :
 * false_branch)
 */
struct ConditionAndBranch {
  ConditionForVar condition_for_var;
  bool branch;
};

std::unordered_map<ir::Var, ir::Var> ConstructForVarReplaceMap(
    const std::vector<ForVarExtent>& lhs_extents,
    const std::vector<ForVarExtent>& rhs_extents) {
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
}

template <typename ExprType>
bool ConditionEqual(const ir::Expr lhs_condition,
                    const ir::Expr rhs_condition) {
  if (lhs_condition.As<ExprType>() && rhs_condition.As<ExprType>()) {
    auto lhs = lhs_condition.As<ExprType>();
    auto rhs = rhs_condition.As<ExprType>();
    if (lhs->a().is_cmp() || rhs->a().is_cmp() || lhs->b().is_cmp() ||
        rhs->b().is_cmp()) {
      return true;
    }
    ir::Expr lhs_equal =
        cinn::common::AutoSimplify(ir::Sub::Make(lhs->a(), rhs->a()));
    ir::Expr rhs_equal =
        cinn::common::AutoSimplify(ir::Sub::Make(lhs->b(), rhs->b()));
    if ((lhs_equal == ir::Expr(0)) && (rhs_equal == ir::Expr(0))) {
      VLOG(6) << "Proved equal conditoin, expr: " << lhs_condition
              << " with expr: " << rhs_condition;
      return true;
    }
  }
  return false;
}

struct GlobalTensorInfoCollector : public ir::IRMutator<Expr*> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  std::unordered_set<std::string> GetEliminateBufferNames() const {
    auto IndiceToExprWithForVar =
        [&](ir::Expr indice,
            const std::unordered_map<ir::Var, ir::Var>& for_var_map)
        -> ir::Expr {
      ir::Expr ret = ir::ir_utils::IRCopy(indice);
      for (const auto& [lhs_var, rhs_var] : for_var_map) {
        ReplaceVarWithExpr(&ret, lhs_var, ir::ir_utils::IRCopy(rhs_var));
      }
      return ret;
    };

    auto IndiceAndExtentEqual =
        [&](const IndicesAndExtent& indice_and_extent1,
            const IndicesAndExtent& indice_and_extent2) -> bool {
      const auto& indice1 = indice_and_extent1.indices;
      const auto& indice2 = indice_and_extent2.indices;
      if (indice1.size() != indice2.size()) return false;

      std::unordered_map<ir::Var, ir::Var> for_var_map =
          ConstructForVarReplaceMap(indice_and_extent1.for_var_extents,
                                    indice_and_extent2.for_var_extents);

      for (size_t i = 0; i < indice1.size(); ++i) {
        ir::Expr lhs = IndiceToExprWithForVar(indice1.at(i), for_var_map);
        ir::Expr rhs = IndiceToExprWithForVar(indice2.at(i), for_var_map);
        if (cinn::common::AutoSimplify(ir::Sub::Make(lhs, rhs)) !=
            ir::Expr(0)) {
          return false;
        }
      }
      return true;
    };

    auto AllIndiceAndExtentEqual =
        [&](const std::vector<IndicesAndExtent>& indice_and_extent) -> bool {
      PADDLE_ENFORCE_GE(
          indice_and_extent.size(),
          2,
          ::common::errors::InvalidArgument(
              "The size of indice_and_extent should greater_equal to 2"));
      for (size_t i = 1; i < indice_and_extent.size(); ++i) {
        if (!IndiceAndExtentEqual(indice_and_extent[0], indice_and_extent[i]))
          return false;
      }
      return true;
    };

    auto IndiceContainsLoad =
        [&](const IndicesAndExtent& indice_and_extent) -> bool {
      for (const auto& index : indice_and_extent.indices) {
        std::set<Expr> load_tensors = ir::ir_utils::CollectLoadTensors(
            index, /*teller=*/[&](const Expr*) -> bool { return true; });
        if (load_tensors.size() > 0) {
          return true;
        }
      }
      return false;
    };

    auto IsGlobalTensorNeedEliminate =
        [&](const std::vector<IndicesAndExtent>& indice_and_extent) -> bool {
      if (indice_and_extent.size() <= 1) return false;
      if (IndiceContainsLoad(indice_and_extent[0])) return false;
      return AllIndiceAndExtentEqual(indice_and_extent);
    };

    auto ConditionContainsLoad = [&](const ir::Expr condition) {
      std::set<Expr> load_tensors = ir::ir_utils::CollectLoadTensors(
          condition, /*teller=*/[&](const Expr*) -> bool { return true; });
      if (load_tensors.size() > 0) {
        return true;
      }
      return false;
    };

    auto ConditionEqualHelper = [&](const ir::Expr lhs_condition,
                                    const ir::Expr rhs_condition) -> bool {
      if (ConditionContainsLoad(lhs_condition) ||
          ConditionContainsLoad(rhs_condition))
        return false;
      if (ConditionEqual<ir::EQ>(lhs_condition, rhs_condition)) return true;
      if (ConditionEqual<ir::NE>(lhs_condition, rhs_condition)) return true;
      if (ConditionEqual<ir::LT>(lhs_condition, rhs_condition)) return true;
      if (ConditionEqual<ir::LE>(lhs_condition, rhs_condition)) return true;
      if (ConditionEqual<ir::GT>(lhs_condition, rhs_condition)) return true;
      if (ConditionEqual<ir::GE>(lhs_condition, rhs_condition)) return true;
      return false;
    };

    auto ConditionWithForVar =
        [&](const ConditionForVar& condition_for_var,
            const std::unordered_map<ir::Var, ir::Var> for_var_map)
        -> ir::Expr {
      ir::Expr condition = condition_for_var.condition;
      ir::Expr ret = ir::ir_utils::IRCopy(condition);
      for (const auto& [var, sb_expr] : var_to_sb_expr_) {
        ReplaceVarWithExpr(&ret, var, ir::ir_utils::IRCopy(sb_expr));
      }
      for (const auto& [lhs_var, rhs_var] : for_var_map) {
        ReplaceVarWithExpr(&ret, lhs_var, ir::ir_utils::IRCopy(rhs_var));
      }
      return ret;
    };

    auto ConditionAndBranchEqual =
        [&](const std::vector<ConditionAndBranch>& condition_and_branch1,
            const std::vector<ConditionAndBranch>& condition_and_branch2)
        -> bool {
      if (condition_and_branch1.size() != condition_and_branch2.size()) {
        return false;
      }
      for (size_t i = 0; i < condition_and_branch1.size(); ++i) {
        if (condition_and_branch1[i].branch !=
            condition_and_branch2[i].branch) {
          return false;
        }
        const ConditionForVar& lhs = condition_and_branch1[i].condition_for_var;
        const ConditionForVar& rhs = condition_and_branch2[i].condition_for_var;
        std::unordered_map<ir::Var, ir::Var> for_var_map =
            ConstructForVarReplaceMap(lhs.for_var_extents, rhs.for_var_extents);
        ir::Expr new_lhs = ConditionWithForVar(lhs, for_var_map);
        ir::Expr new_rhs = ConditionWithForVar(rhs, for_var_map);
        if (!ConditionEqualHelper(new_lhs, new_rhs)) {
          return false;
        }
      }
      return true;
    };

    auto AllConditionAndBranchEqual =
        [&](const std::string buffer_name) -> bool {
      if (buffer_to_condition_and_branch_.find(buffer_name) ==
          buffer_to_condition_and_branch_.end()) {
        return true;
      }
      const std::vector<std::vector<ConditionAndBranch>>
          condition_and_branches =
              buffer_to_condition_and_branch_.at(buffer_name);
      // There are only 0 or 1 conditions, no need to check
      // if the elements are equal.
      if (condition_and_branches.size() <= 1) {
        return true;
      }
      PADDLE_ENFORCE_GE(
          condition_and_branches.size(),
          2,
          ::common::errors::InvalidArgument(
              "The size of condition_and_branch should greater_equal to 2"));
      for (size_t i = 1; i < condition_and_branches.size(); ++i) {
        if (!ConditionAndBranchEqual(condition_and_branches[0],
                                     condition_and_branches[i])) {
          return false;
        }
      }
      return true;
    };

    std::unordered_set<std::string> global_buffer_name;
    for (const auto& [buffer_name, indice_and_extent] :
         buffer_to_indice_and_extent_) {
      // For buffers disobey SSA principle, we don't substitute them.
      if (global_store_buffer_names_.find(buffer_name) !=
          global_store_buffer_names_.end()) {
        continue;
      }
      if (!AllConditionAndBranchEqual(buffer_name)) {
        VLOG(6)
            << "Buffer's condition and branch not equal, use global buffer: "
            << buffer_name;
        continue;
      }
      if (IsGlobalTensorNeedEliminate(indice_and_extent)) {
        global_buffer_name.insert(buffer_name);
      }
    }
    return global_buffer_name;
  }

 private:
  void Visit(const ir::ScheduleBlockRealize* op, ir::Expr* expr) override {
    const auto* sbr_node = expr->As<ir::ScheduleBlockRealize>();
    CHECK(sbr_node);
    const auto& iter_values = sbr_node->iter_values;
    const auto* sb_node = sbr_node->schedule_block.As<ir::ScheduleBlock>();
    const auto& iter_vars = sb_node->iter_vars;
    PADDLE_ENFORCE_EQ(
        iter_values.size(),
        iter_vars.size(),
        ::common::errors::InvalidArgument(
            "The size of iter_values should equal to the size of iter_vars, as "
            "they comes from the same ScheduleBlockRealize"));

    for (std::size_t i = 0; i < iter_values.size(); ++i) {
      var_to_sb_expr_[iter_vars[i]] = iter_values[i];
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::For>();
    CHECK(node);
    for_var_extents_.push_back(
        {node->loop_var, ir::ir_utils::IRCopy(node->extent)});
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::Load* op, ir::Expr* expr) override {
    auto RecordCondition = [&](const std::string buffer_name) {
      if (condition_non_buffer_names_.find(buffer_name) !=
          condition_non_buffer_names_.end()) {
        return;
      }
      // Indicate no condition constraints.
      if (current_condition_and_branches_.empty()) {
        condition_non_buffer_names_.insert(buffer_name);
        buffer_to_condition_and_branch_[buffer_name].clear();
      } else {
        buffer_to_condition_and_branch_[buffer_name].push_back(
            current_condition_and_branches_);
      }
    };

    auto* node = expr->As<ir::Load>();
    CHECK(node);
    const auto& load_buffer = node->tensor.as_tensor_ref()->buffer;
    if (load_buffer->memory_type == ir::MemoryType::Heap) {
      std::vector<ir::Expr> tensor_indices;
      for (const auto& indice : node->indices) {
        ir::Expr new_indice = ir::ir_utils::IRCopy(indice);
        for (const auto& [var, sb_expr] : var_to_sb_expr_) {
          ReplaceVarWithExpr(&new_indice, var, ir::ir_utils::IRCopy(sb_expr));
        }
        tensor_indices.push_back(new_indice);
      }
      buffer_to_indice_and_extent_[load_buffer->name].push_back(
          {tensor_indices, for_var_extents_});
      RecordCondition(load_buffer->name);
    }
  }

  void Visit(const ir::Store* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    CHECK(node);
    const auto& store_buffer = node->tensor.as_tensor_ref()->buffer;
    if (store_buffer->memory_type == ir::MemoryType::Heap) {
      global_store_buffer_names_.insert(store_buffer->name);
    }
  }

  void Visit(const ir::Select* op, ir::Expr* expr) override {
    auto node = expr->As<ir::Select>();
    // The conditional expression does not affect itself,
    // but is constrained by the previous conditions, so that
    // push element after condition get visited.
    ir::IRMutator<>::Visit(&node->condition, &node->condition);
    current_condition_and_branches_.push_back(
        {{node->condition, for_var_extents_}, true});
    ir::IRMutator<>::Visit(&node->true_value, &node->true_value);
    current_condition_and_branches_.back().branch = false;
    ir::IRMutator<>::Visit(&node->false_value, &node->false_value);
    current_condition_and_branches_.pop_back();
  }

  std::vector<ForVarExtent> for_var_extents_;
  std::unordered_map<ir::Var, ir::Expr> var_to_sb_expr_;
  std::unordered_map<std::string, std::vector<IndicesAndExtent>>
      buffer_to_indice_and_extent_;
  std::unordered_set<std::string> global_store_buffer_names_;
  std::vector<ConditionAndBranch> current_condition_and_branches_;
  std::unordered_map<std::string, std::vector<std::vector<ConditionAndBranch>>>
      buffer_to_condition_and_branch_;
  std::unordered_set<std::string> condition_non_buffer_names_;
};

struct CommonGlobalMemoryEliminator : public ir::IRMutator<Expr*> {
  CommonGlobalMemoryEliminator(
      const std::unordered_set<std::string>& eliminate_buffer_names)
      : eliminate_buffer_names_(eliminate_buffer_names) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Select* op, ir::Expr* expr) override {
    auto node = expr->As<ir::Select>();
    ir::IRMutator<>::Visit(&node->condition, &node->condition);
    // When eliminating, extents is not necessary.
    current_condition_and_branches_.push_back({{node->condition, {}}, true});
    ir::IRMutator<>::Visit(&node->true_value, &node->true_value);
    current_condition_and_branches_.back().branch = false;
    ir::IRMutator<>::Visit(&node->false_value, &node->false_value);
  }

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
    if (eliminate_buffer_names_.count(buffer_name) == 0) {
      return;
    }

    if (global_buffer_to_local_buffer_.count(buffer_name) == 0) {
      InsertLocalTensorBlock(node, buffer_name);
    }
    SubstituteGlobalTensor(node, buffer_name);
  }

  ir::Expr MakeConditionStore(int index, ir::Expr load_expr) {
    if (index >= current_condition_and_branches_.size()) {
      return load_expr;
    }
    const ConditionAndBranch condition_and_branch =
        current_condition_and_branches_[index];
    const ir::Expr sub_expr = MakeConditionStore(index + 1, load_expr);
    ir::Expr dummy_expr = ir::Zero(sub_expr.type());
    VLOG(10) << "Origin type: " << sub_expr.type()
             << " Dummy type: " << dummy_expr.type();
    if (condition_and_branch.branch) {
      return ir::Select::Make(condition_and_branch.condition_for_var.condition,
                              sub_expr,
                              dummy_expr);
    } else {
      return ir::Select::Make(condition_and_branch.condition_for_var.condition,
                              dummy_expr,
                              sub_expr);
    }
    PADDLE_THROW(phi::errors::Fatal(
        "Dead code. Fail to make condition store for local buffer."));
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
    constexpr int kmake_condition_store_start = 0;
    ir::Expr condition_store =
        MakeConditionStore(kmake_condition_store_start, ir::Expr(load_node));
    ir::Expr new_body =
        ir::Store::Make(new_tensor,
                        ir::ir_utils::IRCopy(condition_store),
                        ir::ir_utils::IRCopy(load_node->indices));
    ir::Expr new_sb = ir::ScheduleBlock::Make(
        sb_node->iter_vars, {}, {}, sb_node->name + "_local", new_body);

    ir::Expr new_sbr = ir::ScheduleBlockRealize::Make(
        ir::ir_utils::IRCopy(current_sbr_->iter_values), new_sb);
    PADDLE_ENFORCE_EQ(
        global_buffer_to_local_buffer_.count(buffer_name),
        0,
        ::common::errors::InvalidArgument(
            "buffer_name %s should not be in global_buffer_to_local_buffer_",
            buffer_name));
    global_buffer_to_local_buffer_[buffer_name] = new_tensor;
    current_block_->stmts.insert(current_block_->stmts.begin(), new_sbr);
  }

  void SubstituteGlobalTensor(ir::Load* load_node,
                              const std::string& buffer_name) {
    PADDLE_ENFORCE_GT(
        global_buffer_to_local_buffer_.count(buffer_name),
        0,
        ::common::errors::InvalidArgument(
            "global_buffer_to_local_buffer_ should contain buffer_name %s",
            buffer_name));
    load_node->tensor = global_buffer_to_local_buffer_[buffer_name];
  }

  std::unordered_set<std::string> eliminate_buffer_names_;
  std::unordered_map<std::string, ir::Expr> global_buffer_to_local_buffer_;
  std::vector<ConditionAndBranch> current_condition_and_branches_;

  ir::Block* current_block_;
  ir::ScheduleBlockRealize* current_sbr_;
};

}  // namespace

void EliminateCommonGlobalMemoryRead(Expr* e) {
  VLOG(4) << "Before EliminateCommonGlobalMemoryRead: \n" << *e;
  GlobalTensorInfoCollector collector;
  collector(e);

  const auto& eliminate_buffer_names = collector.GetEliminateBufferNames();

  CommonGlobalMemoryEliminator eliminator(eliminate_buffer_names);
  eliminator(e);
  VLOG(4) << "After EliminateCommonGlobalMemoryRead: \n" << *e;
}

}  // namespace optim
}  // namespace cinn
