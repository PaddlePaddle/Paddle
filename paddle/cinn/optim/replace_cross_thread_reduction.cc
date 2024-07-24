// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

/**
 * This file implements the strategy to remove the unnecessary nested block.
 */
#pragma once
#include "paddle/cinn/optim/replace_cross_thread_reduction.h"
#include <vector>

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/hlir/pe/reduction.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace optim {
namespace {

struct BufferCmp {
  bool operator()(const ir::Buffer& a, const ir::Buffer& b) const {
    if (a->name == b->name) return false;
    return true;
  }
};

thread_local std::set<ir::Buffer, BufferCmp> shm_buffer_;
struct CrossThreadReductionReplacer : public ir::IRMutator<> {
  void operator()(ir::Expr* expr) { Visit(expr); }

 private:
  bool CanReplace(const ir::ScheduleBlockRealize* block_realize) {
    const ir::ScheduleBlock* schedule_block =
        block_realize->schedule_block.As<ir::ScheduleBlock>();

    PADDLE_ENFORCE_NOT_NULL(
        schedule_block,
        phi::errors::PreconditionNotMet(
            "The schedule block pointer in CanReplace must not be null."));

    if (block_realize->schedule_block.As<ir::ScheduleBlock>()->name.substr(
            0, 4) == "root") {
      return false;
    }

    const std::vector<ir::Expr>& iter_values = block_realize->iter_values;
    const std::vector<ir::Var>& iter_vars = schedule_block->iter_vars;
    ir::Expr body = schedule_block->body;

    std::unordered_set<std::string> reduce_var_names;
    for (int i = 0; i < iter_values.size(); ++i) {
      if (!iter_vars[i]->is_reduce_axis) {
        continue;
      }
      ir::ir_utils::CollectIRNodesWithoutTensor(
          iter_values[i], [&](const ir::Expr* x) {
            if (x->as_var()) {
              reduce_var_names.insert(x->as_var()->name);
            }

            return false;
          });
    }

    auto IsThreadBindOnReduceAxis = [&](const ir::For* for_node) {
      return reduce_var_names.count(for_node->loop_var->name) > 0 &&
             for_node->is_gpu_thread_binded();
    };

    std::vector<int> thread_binded_reduce_loop_indices;
    bool is_thread_binded_inner_loop = false;
    for (int i = 0; i < cur_loops_.size(); ++i) {
      if (is_thread_binded_inner_loop ||
          IsThreadBindOnReduceAxis(cur_loops_[i].As<ir::For>())) {
        if (ir::GetLoopExtent(cur_loops_[i]) > 1024) {
          return false;
        }

        is_thread_binded_inner_loop = true;
        thread_binded_reduce_loop_indices.push_back(i);
      }
    }
    if (thread_binded_reduce_loop_indices.size() == 0 ||
        thread_binded_reduce_loop_indices.back() != cur_loops_.size() - 1) {
      return false;
    }
    for (int i = 1; i < thread_binded_reduce_loop_indices.size(); ++i) {
      if (thread_binded_reduce_loop_indices[i - 1] + 1 !=
          thread_binded_reduce_loop_indices[i]) {
        return false;
      }
    }

    return true;
  }

  int GetBlockSize() const {
    int block_size = 1;
    for (auto& loop : cur_loops_) {
      if (loop->as<ir::For>()->is_gpu_thread_binded()) {
        block_size *= ir::GetLoopExtent(loop);
      }
    }
    return block_size;
  }

  template <typename OpT>
  void ReplaceByContinuousReduceExternCall(ir::Expr* store, bool return_warp) {
    auto* node = store->As<ir::Store>()->value.As<OpT>();
    CHECK(node);
    auto& operand = node->b();
    std::string reduce_func_name = hlir::pe::CrossThreadReduceExternalFuncName(
        store->As<ir::Store>()->value, operand.template As<ir::Load>()->tensor);
    auto tmp_dtype =
        operand.template As<ir::Load>()->tensor.as_tensor()->type();
    auto tmp_buffer = ir::_Buffer_::Make(
        "shm32_" + hlir::pe::Type2StrForReduce(tmp_dtype) + "_reduce",
        {ir::Expr(32)});
    tmp_buffer->dtype = tmp_dtype;
    tmp_buffer->memory_type = ir::MemoryType::GPUShared;
    shm_buffer_.insert(tmp_buffer);
    store->As<ir::Store>()->value = lang::CallExtern(
        reduce_func_name, {node->b(), tmp_buffer, ir::Expr(return_warp)});
  }

  template <typename OpT>
  void ReplaceByDiscreteReduceExternCall(ir::Expr* store) {
    auto* node = store->As<ir::Store>()->value.As<OpT>();
    CHECK(node);
    auto& operand = node->b();
    std::string reduce_func_name = hlir::pe::DiscreteReduceExternalFuncName(
        store->As<ir::Store>()->value, operand.template As<ir::Load>()->tensor);
    auto tmp_dtype =
        operand.template As<ir::Load>()->tensor.as_tensor()->type();
    auto tmp_buffer = ir::_Buffer_::Make(
        "shm32_" + hlir::pe::Type2StrForReduce(tmp_dtype) + "_reduce",
        {ir::Expr(GetBlockSize())});
    tmp_buffer->dtype = tmp_dtype;
    tmp_buffer->memory_type = ir::MemoryType::GPUShared;
    shm_buffer_.insert(tmp_buffer);
    store->As<ir::Store>()->value =
        lang::CallExtern(reduce_func_name, {node->b(), tmp_buffer});
  }

  template <typename OpT>
  void ReplaceByReduceExternCall(ir::Expr* store,
                                 const ir::ReduceMethod& method) {
    std::visit(cinn::adt::match{
                   [&](const ir::NoneReduceMethod&) {
                     ReplaceByContinuousReduceExternCall<OpT>(store, false);
                   },
                   [&](const ir::WarpReduceMethod&) {
                     ReplaceByContinuousReduceExternCall<OpT>(store, true);
                   },
                   [&](const ir::BlockReduceMethod&) {
                     ReplaceByContinuousReduceExternCall<OpT>(store, false);
                   },
                   [&](const ir::DiscreteReduceMethod&) {
                     ReplaceByDiscreteReduceExternCall<OpT>(store);
                   }},
               method);
  }

  void Visit(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::_LoweredFunc_* expr, ir::Expr* op) override {
    ir::IRMutator<>::Visit(expr, op);
    if (std::find_if(op->as_lowered_func()->temp_bufs.begin(),
                     op->as_lowered_func()->temp_bufs.end(),
                     [&](const ir::Buffer& buf) -> bool {
                       for (auto& tmp_buf : shm_buffer_) {
                         if (buf->name == tmp_buf->name) return true;
                       }
                       return false;
                     }) == op->as_lowered_func()->temp_bufs.end())
      op->as_lowered_func()->temp_bufs.insert(
          op->as_lowered_func()->temp_bufs.end(),
          shm_buffer_.begin(),
          shm_buffer_.end());
    shm_buffer_.clear();
  }

  void Visit(const ir::ScheduleBlockRealize* expr, ir::Expr* op) override {
    if (!CanReplace(expr)) {
      VLOG(6) << "Can't replace cross thread reduction: " << *op;
      IRMutator::Visit(expr, op);
      return;
    }
    VLOG(6) << "Can replace cross thread reduction: " << *op;

    const ir::ScheduleBlock* schedule_block =
        expr->schedule_block.As<ir::ScheduleBlock>();
    PADDLE_ENFORCE_NOT_NULL(
        schedule_block,
        phi::errors::PreconditionNotMet(
            "The schedule block pointer in Visit must not be null."));
    ir::Expr original_update_body = schedule_block->body;
    ir::Expr original_update_stmt;
    CHECK(original_update_body.As<ir::Block>() ||
          original_update_body.As<ir::Store>());
    if (original_update_body.As<ir::Block>()) {
      PADDLE_ENFORCE_EQ(
          original_update_body.As<ir::Block>()->stmts.size(),
          1,
          phi::errors::InvalidArgument(
              "The size of stmts is incorrect."
              "Expected size is 1, but receive %d.",
              original_update_body.As<ir::Block>()->stmts.size()));
      original_update_stmt = original_update_body.As<ir::Block>()->stmts[0];
    } else if (original_update_body.As<ir::Store>()) {
      original_update_stmt = original_update_body;
    }

#define REPLACE_TO_EXTERNAL_CALL(Op)                              \
  if (original_update_stmt.As<ir::Store>()->value.As<Op>()) {     \
    ReplaceByReduceExternCall<Op>(&original_update_stmt,          \
                                  schedule_block->reduce_method); \
  }

    REPLACE_TO_EXTERNAL_CALL(ir::Add)
    REPLACE_TO_EXTERNAL_CALL(ir::Mul)
    REPLACE_TO_EXTERNAL_CALL(ir::Max)
    REPLACE_TO_EXTERNAL_CALL(ir::Min)
    REPLACE_TO_EXTERNAL_CALL(ir::And)
    REPLACE_TO_EXTERNAL_CALL(ir::Or)
#undef REPLACE_TO_EXTERNAL_CALL

    VLOG(6) << "Replace cross thread reduction: " << *op;

    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, ir::Expr* op) override {
    cur_loops_.push_back(*op);
    IRMutator::Visit(expr, op);
    cur_loops_.pop_back();
  }

 private:
  std::vector<ir::Expr> cur_loops_;
};

}  // namespace

void ReplaceCrossThreadReduction(Expr* e) { CrossThreadReductionReplacer()(e); }

}  // namespace optim
}  // namespace cinn
