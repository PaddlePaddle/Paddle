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
#include "paddle/cinn/common/integer_set.h"
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

std::unordered_map<ir::Var, ir::Var> ConstructForVarReplaceMap(
    const std::vector<ForVarExtent>& lhs_extents,
    const std::vector<ForVarExtent>& rhs_extents) {
  std::unordered_map<ir::Var, ir::Var> ret;
  std::unordered_set<std::size_t> visited_rhs_index;
  for (const auto& [lhs_var, lhs_extent] : lhs_extents) {
    for (std::size_t i = 0; i < rhs_extents.size(); ++i) {
      const auto& [rhs_var, rhs_extent] = rhs_extents[i];
      if (optim::ArithSimplify(ir::Sub::Make(lhs_extent, rhs_extent)) ==
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
        if (optim::ArithSimplify(ir::Sub::Make(lhs, rhs)) != ir::Expr(0)) {
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
      if (contains_select_) return false;
      return AllIndiceAndExtentEqual(indice_and_extent);
    };

    auto BufferSizeContainsSymbolic = [&](const ir::Expr& buffer_size) -> bool {
      bool has_symbolic = false;
      ir::ir_utils::CollectIRNodes(buffer_size, [&](const ir::Expr* x) {
        if (x->as_var() && x->as_var()->is_symbolic_constant) {
          has_symbolic = true;
        }
        return false;
      });
      return has_symbolic;
    };

    auto GetIterVarNames =
        [&](const std::vector<ir::Expr>& indices) -> std::set<std::string> {
      std::set<std::string> iter_var_names;
      for (const ir::Expr& e : indices) {
        ir::ir_utils::CollectIRNodes(e, [&](const ir::Expr* x) {
          if (x->as_var() && !x->as_var()->is_symbolic_constant) {
            iter_var_names.insert(x->as_var()->name);
          }
          return false;
        });
      }
      return iter_var_names;
    };

    auto CalculateBufferSize =
        [&](const std::vector<ir::Expr>& indices) -> ir::Expr {
      ir::Expr buffer_size(1);
      std::set<std::string> iter_var_names = GetIterVarNames(indices);
      for (const auto& iter_var_name : iter_var_names) {
        if (iter_var_name_to_extent_.find(iter_var_name) ==
            iter_var_name_to_extent_.end()) {
          continue;
        }
        VLOG(6) << "Iter var name: " << iter_var_name << " with extent: "
                << iter_var_name_to_extent_.at(iter_var_name);
        buffer_size = optim::ArithSimplify(ir::Mul::Make(
            buffer_size, iter_var_name_to_extent_.at(iter_var_name)));
      }
      return buffer_size;
    };

    auto LocalBufferSizeLimit =
        [&](const std::unordered_set<std::string>& global_buffer_name) -> bool {
      ir::Expr size(0);
      for (const auto& name : global_buffer_name) {
        const std::vector<IndicesAndExtent>& indices_and_extent =
            buffer_to_indice_and_extent_.at(name);
        const ir::Expr buffer_size =
            CalculateBufferSize(indices_and_extent[0].indices);
        VLOG(6) << "Global buffer name: " << name
                << " with size: " << buffer_size;
        size = optim::ArithSimplify(ir::Add::Make(size, buffer_size));
      }
      if (BufferSizeContainsSymbolic(size)) {
        VLOG(6) << "Local buffer size contains symbolic: " << size;
        return true;
      }
      VLOG(6) << "Total buffer size: " << size;
      common::cas_intervals_t var_intervals;
      common::SymbolicExprAnalyzer analyzer(var_intervals);
      std::optional<bool> prove_gt = analyzer.ProveGT(size, ir::Expr(8));
      return prove_gt.value_or(false);
    };

    std::unordered_set<std::string> global_buffer_name;
    for (const auto& [buffer_name, indice_and_extent] :
         buffer_to_indice_and_extent_) {
      // For buffers disobey SSA principle, we don't substitute them.
      if (global_store_buffer_names_.find(buffer_name) !=
          global_store_buffer_names_.end()) {
        continue;
      }
      if (IsGlobalTensorNeedEliminate(indice_and_extent)) {
        global_buffer_name.insert(buffer_name);
      }
    }
    // When local buffer size too large, it will cause
    // out of memory error, use global buffer instead.
    // Fuse for loop will relax this constraints.
    if (LocalBufferSizeLimit(global_buffer_name)) {
      VLOG(6) << "Local buffer size too large or contains symbolic var, use "
                 "global buffer instead.\n";
      global_buffer_name.clear();
    }
    return global_buffer_name;
  }

 private:
  void Visit(const ir::ScheduleBlockRealize* op, ir::Expr* expr) override {
    const auto* sbr_node = expr->As<ir::ScheduleBlockRealize>();
    PADDLE_ENFORCE_NOT_NULL(
        sbr_node,
        ::common::errors::InvalidArgument(
            "The input expr should be a ScheduleBlockRealize"));
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
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument("The input expr should be a For"));
    for_var_extents_.push_back(
        {node->loop_var, ir::ir_utils::IRCopy(node->extent)});
    if (!node->is_binded()) {
      iter_var_name_to_extent_[node->loop_var->name] = node->extent;
    }
    ir::IRMutator<>::Visit(op, expr);
    for_var_extents_.pop_back();
  }

  void Visit(const ir::Load* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Load>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument("The input expr should be a Load"));

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
    }
  }

  void Visit(const ir::Store* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument("The input expr should be a Store"));
    const auto& store_buffer = node->tensor.as_tensor_ref()->buffer;
    if (store_buffer->memory_type == ir::MemoryType::Heap) {
      global_store_buffer_names_.insert(store_buffer->name);
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Select* op, ir::Expr* expr) override {
    contains_select_ = true;
    ir::IRMutator<>::Visit(op, expr);
  }

  std::vector<ForVarExtent> for_var_extents_;
  std::unordered_map<ir::Var, ir::Expr> var_to_sb_expr_;
  std::unordered_map<std::string, ir::Expr> iter_var_name_to_extent_;
  std::unordered_map<std::string, std::vector<IndicesAndExtent>>
      buffer_to_indice_and_extent_;
  std::unordered_set<std::string> global_store_buffer_names_;
  bool contains_select_ = false;
};

struct CommonGlobalMemoryEliminator : public ir::IRMutator<Expr*> {
  CommonGlobalMemoryEliminator(
      const std::unordered_set<std::string>& eliminate_buffer_names)
      : eliminate_buffer_names_(eliminate_buffer_names) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Block* op, Expr* expr) override {
    auto* node = expr->As<ir::Block>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument("The input expr should be a Block"));
    current_block_ = node;
    IRMutator<>::Visit(op, expr);

    // Insert buffer declare after visit current block.
    if (block_to_insert_stmts_.find(node) != block_to_insert_stmts_.end()) {
      const std::vector<ir::Expr>& insert_schedule_blocks =
          block_to_insert_stmts_[node];
      for (const ir::Expr& block : insert_schedule_blocks) {
        node->stmts.insert(node->stmts.begin(), block);
      }
    }
  }

  void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override {
    auto* node = expr->As<ir::ScheduleBlockRealize>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument(
            "The input expr should be a ScheduleBlockRealize"));
    current_sbr_ = node;
    if (current_block_) {
      insert_block_ = current_block_;
    }
    IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load* op, Expr* expr) override {
    auto* node = expr->As<ir::Load>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument("The input expr should be a Load"));
    const auto& buffer_name = node->tensor.as_tensor_ref()->buffer->name;
    if (eliminate_buffer_names_.count(buffer_name) == 0) {
      return;
    }

    if (global_buffer_to_local_buffer_.count(buffer_name) == 0) {
      InsertLocalTensorBlock(node, buffer_name);
    }
    SubstituteGlobalTensor(node, buffer_name);
  }

  void InsertLocalTensorBlock(ir::Load* load_node,
                              const std::string& buffer_name) {
    ir::Expr sb = ir::ir_utils::IRCopy(current_sbr_->schedule_block);
    ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
    PADDLE_ENFORCE_NOT_NULL(
        sb_node,
        ::common::errors::InvalidArgument(
            "The input expr should be a ScheduleBlockRealize"));
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
    PADDLE_ENFORCE_EQ(
        global_buffer_to_local_buffer_.count(buffer_name),
        0,
        ::common::errors::InvalidArgument(
            "buffer_name %s should not be in global_buffer_to_local_buffer_",
            buffer_name));
    global_buffer_to_local_buffer_[buffer_name] = new_tensor;

    PADDLE_ENFORCE_NOT_NULL(
        insert_block_,
        ::common::errors::InvalidArgument("insert block CAN NOT be nullptr"));
    block_to_insert_stmts_[insert_block_].push_back(new_sbr);
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
  std::unordered_map<ir::Block*, std::vector<ir::Expr>> block_to_insert_stmts_;

  ir::Block* current_block_{nullptr};
  ir::Block* insert_block_{nullptr};
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
