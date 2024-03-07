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

#include "paddle/cinn/optim/eliminate_common_global_var.h"

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

struct GlobalVarInfoCollector : public ir::IRMutator<Expr*> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  std::unordered_set<std::string> GetEliminateVarNames() const {
    auto IndiceAndExtentEqual =
        [](const IndicesAndExtent& indice_and_extent1,
           const IndicesAndExtent& indice_and_extent2) -> bool {
      const auto& indice1 = indice_and_extent1.indices;
      const auto& indice2 = indice_and_extent2.indices;
      // VLOG(-1) << "DEBUG indice1:\n" << indice1;
      // VLOG(-1) << "DEBUG indice2:\n" << indice2;
      return true;
      // [] Check logic here
      if (indice1.size() != indice2.size()) return false;
      for (size_t i = 0; i < indice1.size(); ++i) {
        if (!ir::ir_utils::IRCompare(indice1.at(i), indice2.at(i), true)) {
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

    auto IsGlobalVarNeedEliminate =
        [&](const std::vector<IndicesAndExtent>& indice_and_extent) -> bool {
      if (indice_and_extent.size() <= 1) return false;
      return AllIndiceAndExtentEqual(indice_and_extent);
    };

    std::unordered_set<std::string> global_var;
    for (const auto& [var_name, indice_and_extent] :
         var_to_indice_and_extent_) {
      if (IsGlobalVarNeedEliminate(indice_and_extent)) {
        global_var.insert(var_name);
      }
    }
    return global_var;
  }

 private:
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
      var_to_indice_and_extent_[load_buffer->name].push_back(
          {node->indices, for_var_extents_});
    }
  }

  std::vector<ForVarExtent> for_var_extents_;
  std::unordered_map<std::string, std::vector<IndicesAndExtent>>
      var_to_indice_and_extent_;
};

struct CommonGlobalVarEliminator : public ir::IRMutator<Expr*> {
  CommonGlobalVarEliminator(
      const std::unordered_set<std::string>& eliminate_var_names)
      : eliminate_var_names_(eliminate_var_names) {}

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
    if (eliminate_var_names_.count(buffer_name) == 0) {
      return;
    }

    if (global_var_to_local_var_.count(buffer_name) == 0) {
      InsertLocalVarBlock(node, buffer_name);
    }
    SubstituteGlobalVar(node, buffer_name);
  }

  void InsertLocalVarBlock(ir::Load* load_node,
                           const std::string& buffer_name) {
    ir::Expr sb = ir::ir_utils::IRCopy(current_sbr_->schedule_block);
    ir::ScheduleBlock* sb_node = sb.As<ir::ScheduleBlock>();
    CHECK(sb_node);

    // [] How to new a Tensor?
    // ir::Expr new_tensor = ir::ir_utils::IRCopy(load_node->tensor);
    const auto& old_tensor = load_node->tensor.as_tensor_ref();
    ir::Expr new_tensor = ir::_Tensor_::Make(old_tensor->name + "_local",
                                             old_tensor->type(),
                                             old_tensor->shape,
                                             old_tensor->domain,
                                             old_tensor->reduce_axis);
    new_tensor.as_tensor_ref()->WithBuffer(
        "local", new_tensor.as_tensor_ref()->name + "_local");
    ir::Expr new_body =
        ir::Store::Make(new_tensor,
                        ir::ir_utils::IRCopy(ir::Expr(load_node)),
                        load_node->indices);
    // [] new_tensor should be write_buffers?
    ir::Expr new_sb = ir::ScheduleBlock::Make(
        sb_node->iter_vars, {}, {}, sb_node->name + "_local", new_body);

    ir::Expr new_sbr =
        ir::ScheduleBlockRealize::Make(current_sbr_->iter_values, new_sb);
    CHECK_EQ(global_var_to_local_var_.count(buffer_name), 0);
    global_var_to_local_var_[buffer_name] = new_tensor;
    current_block_->stmts.insert(current_block_->stmts.begin(), new_sbr);
  }

  void SubstituteGlobalVar(ir::Load* load_node,
                           const std::string& buffer_name) {
    CHECK_GT(global_var_to_local_var_.count(buffer_name), 0);
    load_node->tensor = global_var_to_local_var_[buffer_name];
    ir::ScheduleBlock* sb_node =
        current_sbr_->schedule_block.As<ir::ScheduleBlock>();
    // sb_node->read_buffers = {global_var_to_local_var_[buffer_name]};
  }

 private:
  std::unordered_set<std::string> eliminate_var_names_;
  std::unordered_map<std::string, ir::Expr> global_var_to_local_var_;

  ir::Block* current_block_;
  ir::ScheduleBlockRealize* current_sbr_;
};

}  // namespace

void EliminateCommonGlobalVar(Expr* e) {
  VLOG(2) << "Before EliminateCommonGlobalVar\n" << *e;

  GlobalVarInfoCollector collector;
  collector(e);

  const auto& eliminate_var_names = collector.GetEliminateVarNames();

  CommonGlobalVarEliminator eliminator(eliminate_var_names);
  eliminator(e);

  VLOG(2) << "After EliminateCommonGlobalVar\n" << *e;
}

}  // namespace optim
}  // namespace cinn
