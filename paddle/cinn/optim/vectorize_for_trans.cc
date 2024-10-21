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

#include "paddle/cinn/optim/vectorize_for_trans.h"

#include <stack>
#include <vector>
#include "paddle/cinn/adt/map_expr.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

namespace {

struct VectorizeForTransMutator : public ir::IRMutator<Expr*> {
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

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

  // void Visit(const ir::Load* op, Expr* expr) override {
  //   auto* node = expr->As<ir::Load>();
  //   PADDLE_ENFORCE_NOT_NULL(
  //       node,
  //       ::common::errors::InvalidArgument("The input expr should be a
  //       Load"));
  //   const auto& buffer_name = node->tensor.as_tensor_ref()->buffer->name;
  //   if (eliminate_buffer_names_.count(buffer_name) == 0) {
  //     return;
  //   }

  //   if (global_buffer_to_local_buffer_.count(buffer_name) == 0) {
  //     InsertLocalTensorForBlock(node, buffer_name);
  //   }
  // }

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();

    const auto& buffer_name = node->tensor.as_tensor_ref()->buffer->name;

    if (in_vectorize_) {
      std::cerr << "buffer name " << buffer_name << std::endl;

      std::cerr << op->value << std::endl;
      std::cerr << op->tensor << std::endl;

      auto load_node = op->value.As<ir::Load>();
      auto load_tensor = load_node->tensor.as_tensor_ref();

      std::string new_vec_load_name =
          "reinterpret_cast<const half2*>(" + load_tensor->name + ")";
      auto data_type =
          Type(Type::type_t::Float, 32, 1, Type::specific_type_t::HALF2);

      // auto new_shape
      ir::Expr new_load_tensor =
          ir::_Tensor_::Make(new_vec_load_name,
                             data_type,
                             ir::ir_utils::IRCopy(load_tensor->shape),
                             ir::ir_utils::IRCopy(load_tensor->domain),
                             load_tensor->reduce_axis);

      // new_load_tensor.as_tensor_ref()->WithBuffer(
      //     "local", new_tensor.as_tensor_ref()->name + "_buffer");

      std::vector<ir::Expr> vec_load_indices;
      for (auto& val : load_node->indices) {
        ir::Expr new_indice = ir::ir_utils::IRCopy(val);
        ReplaceVarWithExpr(&new_indice, loop_var, ir::Expr(0l));
        vec_load_indices.push_back(new_indice);
      }

      ir::Expr vec_load = ir::Load::Make(new_load_tensor, vec_load_indices);

      ir::Expr vec_var = ir::_Var_::Make(
          "var_vec",
          Type(Type::type_t::Float, 32, 1, Type::specific_type_t::HALF2));
      std::cerr << "vec var " << vec_var << std::endl;
      std::cerr << "vec var type " << vec_var.type() << std::endl;
      ir::Expr load_and_assign = ir::Let::Make(vec_var, vec_load);

      std::cerr << "load and assign \n" << load_and_assign << std::endl;

      std::vector<ir::Expr> vec_store_indices_0;
      std::vector<ir::Expr> vec_store_indices_1;

      for (auto& val : node->indices) {
        ir::Expr new_indice_0 = ir::ir_utils::IRCopy(val);
        ReplaceVarWithExpr(&new_indice_0, loop_var, ir::Expr(0l));

        ir::Expr new_indice_1 = ir::ir_utils::IRCopy(val);
        ReplaceVarWithExpr(&new_indice_1, loop_var, ir::Expr(1l));

        vec_store_indices_0.push_back(new_indice_0);
        vec_store_indices_1.push_back(new_indice_1);
      }

      ir::Expr local_store_0 =
          ir::Store::Make(node->tensor,
                          ir::Cast::Make(node->tensor.type(),
                                         ir::StructElement::Make(vec_var, "x")),
                          vec_store_indices_0);

      ir::Expr local_store_1 =
          ir::Store::Make(node->tensor,
                          ir::Cast::Make(node->tensor.type(),
                                         ir::StructElement::Make(vec_var, "y")),
                          vec_store_indices_1);

      std::cerr << "load store 0 " << local_store_0 << "\n"
                << local_store_1 << std::endl;

      block_to_insert_stmts_[insert_vectorize_block_].push_back(local_store_1);

      block_to_insert_stmts_[insert_vectorize_block_].push_back(local_store_0);

      block_to_insert_stmts_[insert_vectorize_block_].push_back(
          load_and_assign);
    }
  }

  void Visit(const ir::For* op, Expr* expr) override {
    if (op->for_type() == ir::ForType::Vectorized) {
      insert_vectorize_block_ = current_block_;
      std::cerr << "loop vars " << op->loop_var << std::endl;
      loop_var = op->loop_var;
      vec_for_ = expr->As<ir::For>();
      in_vectorize_ = true;
    }

    IRMutator<>::Visit(op, expr);

    in_vectorize_ = false;
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
    // block_to_insert_stmts_[insert_block_].push_back(new_sbr);
  }

  void InsertLocalTensorForBlock(ir::Load* load_node,
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

    ir::Expr new_for = ir::For::Make(vec_for_->loop_var,
                                     vec_for_->min,
                                     vec_for_->extent,
                                     ir::ForType::Vectorized,
                                     ir::DeviceAPI::UNK,
                                     new_sbr,
                                     ir::VectorizeInfo(1, 2));

    block_to_insert_stmts_[insert_vectorize_block_].push_back(new_for);
  }

  std::unordered_set<std::string> eliminate_buffer_names_;
  std::unordered_map<std::string, ir::Expr> global_buffer_to_local_buffer_;
  std::unordered_map<ir::Block*, std::vector<ir::Expr>> block_to_insert_stmts_;

  ir::Block* current_block_{nullptr};
  ir::Block* insert_block_{nullptr};
  ir::Block* insert_vectorize_block_{nullptr};
  int vectorize_size{0};
  ir::Var loop_var;
  ir::For* vec_for_{nullptr};
  ir::ScheduleBlockRealize* current_sbr_;
  bool in_vectorize_{false};
};
}  // namespace

void VectorizeForTrans(Expr* expr) {
  VectorizeForTransMutator collector;

  std::cerr << "before vectorize for trans " << *expr << std::endl;
  collector(expr);

  std::cerr << "after vectorize for trans " << *expr << std::endl;
}

}  // namespace optim
}  // namespace cinn
