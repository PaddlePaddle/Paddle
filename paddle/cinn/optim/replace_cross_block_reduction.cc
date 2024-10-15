// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/replace_cross_block_reduction.h"
#include <vector>

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/hlir/pe/reduction.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace optim {
namespace {

ir::Expr CalcBufferSizeInBytes(const ir::Buffer& buffer) {
  const ir::Expr numel = buffer->SymbolicNumel();
  return common::AutoSimplify(numel * buffer->dtype.bytes());
}

std::unordered_set<std::string> GetReduceVarNames(
    const ir::ScheduleBlockRealize* block_realize) {
  const ir::ScheduleBlock* schedule_block =
      block_realize->schedule_block.As<ir::ScheduleBlock>();
  const std::vector<ir::Expr>& iter_values = block_realize->iter_values;
  const std::vector<ir::Var>& iter_vars = schedule_block->iter_vars;

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
  return reduce_var_names;
}

ir::Expr GetRightOperand(const ir::Expr& expr) {
#define GET_RIGHT_OPERAND(OpT)  \
  if (expr.As<OpT>()) {         \
    return expr.As<OpT>()->b(); \
  }

  GET_RIGHT_OPERAND(ir::Add);
  GET_RIGHT_OPERAND(ir::Mul);
  GET_RIGHT_OPERAND(ir::Max);
  GET_RIGHT_OPERAND(ir::Min);
  GET_RIGHT_OPERAND(ir::And);
  GET_RIGHT_OPERAND(ir::Or);

#undef GET_RIGHT_OPERAND
  PADDLE_THROW(
      ::common::errors::InvalidArgument("Not a supported reduce op: %s", expr));
}

struct CrossBlockReductionReplacer : public ir::IRMutator<> {
  void operator()(ir::Expr* expr) { Visit(expr); }

 private:
  bool IsGridReduce(const ir::ScheduleBlockRealize* block_realize) {
    if (cur_loops_.empty()) {
      return false;
    }
    auto* innermost_loop = cur_loops_.back();
    if (!innermost_loop->is_gpu_block_binded()) {
      return false;
    }
    const std::unordered_set<std::string> reduce_var_names =
        GetReduceVarNames(block_realize);
    return reduce_var_names.count(innermost_loop->loop_var->name) > 0;
  }

  void InsertTempSpaceToFuncArgs(ir::_LoweredFunc_* func_node,
                                 const ir::Buffer& buffer,
                                 bool need_zero_init) {
    // insert the temp space after the last tensor argument and before the
    // first scalar argument
    auto insert_pos =
        std::find_if(func_node->args.begin(),
                     func_node->args.end(),
                     [](const ir::Argument& arg) { return arg.is_var(); });

    int arg_idx = std::distance(func_node->args.begin(), insert_pos);
    func_node->temp_spaces.emplace_back(
        CalcBufferSizeInBytes(buffer), arg_idx, need_zero_init);

    ir::Argument temp_space_arg(buffer, ir::Argument::IO::kOutput);
    func_node->args.insert(insert_pos, temp_space_arg);
  }

  void ConvertHeapBuffersToFuncArgs(ir::_LoweredFunc_* func_node) {
    std::vector<ir::Buffer> global_bufs;
    std::vector<ir::Buffer> local_bufs;

    for (auto& buf : func_node->temp_bufs) {
      if (buf->memory_type == ir::MemoryType::Heap) {
        global_bufs.push_back(buf);
      } else {
        local_bufs.push_back(buf);
      }
    }

    PADDLE_ENFORCE_LE(global_bufs.size(),
                      1UL,
                      ::common::errors::PreconditionNotMet(
                          "Currently supports at most one global buffer."));

    for (auto& buf : global_bufs) {
      InsertTempSpaceToFuncArgs(func_node, buf, false);
    }
    func_node->temp_bufs = local_bufs;
  }

  ir::Tensor CreateLastBlockDoneTensor() {
    if (is_done_tensor_.defined()) {
      return is_done_tensor_;
    }
    const std::string name = "is_last_block_done";
    const std::vector<ir::Expr> shape = {};
    is_done_tensor_ = ir::_Tensor_::Make(name, common::Bool(), shape, shape);
    is_done_tensor_->WithBuffer("local", "_" + name + "_temp_buffer");
    return is_done_tensor_;
  }

  ir::Expr GetBlockBindedSpatialLoopExtend(
      const ir::ScheduleBlockRealize* block_realize) {
    const std::unordered_set<std::string> reduce_var_names =
        GetReduceVarNames(block_realize);
    std::vector<ir::Expr> loop_extends;
    for (auto* for_node : cur_loops_) {
      if (reduce_var_names.count(for_node->loop_var->name) == 0 &&
          for_node->is_gpu_block_binded()) {
        loop_extends.push_back(for_node->extent);
      }
    }
    PADDLE_ENFORCE_EQ(
        loop_extends.size(),
        1UL,
        ::common::errors::PreconditionNotMet(
            "There should be exactly one spatial loop binded on gpu block."));
    return loop_extends[0];
  }

  ir::Expr GetThreadBindedSpatialLoopExtend(
      const ir::ScheduleBlockRealize* block_realize) {
    const std::unordered_set<std::string> reduce_var_names =
        GetReduceVarNames(block_realize);
    std::vector<ir::Expr> loop_extends;
    for (auto* for_node : cur_loops_) {
      if (reduce_var_names.count(for_node->loop_var->name) == 0 &&
          for_node->is_gpu_thread_binded()) {
        loop_extends.push_back(for_node->extent);
      }
    }
    PADDLE_ENFORCE_LE(
        loop_extends.size(),
        1UL,
        ::common::errors::PreconditionNotMet(
            "There could be at most one spatial loop binded on gpu thread."));
    if (loop_extends.empty()) {
      return ir::Expr(1);
    }
    return loop_extends[0];
  }

  ir::Expr CreateSemaphoreUpdateStmt(
      const std::vector<ir::Expr>& semaphore_shape) {
    const std::string name = "semaphore";
    ir::Tensor semaphore = ir::_Tensor_::Make(
        name, common::Int(32), semaphore_shape, semaphore_shape);
    semaphore->WithBuffer("global", "_" + name);
    semaphore_buffer_ = semaphore->buffer;
    ir::Expr update_semaphore =
        lang::CallExtern("cinn_grid_reduce_update_semaphore", {semaphore});
    ir::Tensor is_done = CreateLastBlockDoneTensor();
    return ir::Store::Make(is_done, update_semaphore, /* indices= */ {});
  }

  ir::Expr WrapInLastBlockDone(ir::Expr* op) {
    ir::Tensor is_done = CreateLastBlockDoneTensor();
    ir::Expr load_is_done = ir::Load::Make(is_done, /* indices= */ {});
    return ir::IfThenElse::Make(load_is_done, *op);
  }

  void ReplaceByGridReduceExternCall(const ir::ScheduleBlock* schedule_block,
                                     const ir::Expr num_spatial_threads) {
    ir::Expr update_stmt = schedule_block->body;
    if (update_stmt.As<ir::Block>()) {
      PADDLE_ENFORCE_EQ(
          update_stmt.As<ir::Block>()->stmts.size(),
          1UL,
          ::common::errors::InvalidArgument(
              "There should be exactly one statment inside schedule_block."));
      update_stmt = update_stmt.As<ir::Block>()->stmts[0];
    }
    PADDLE_ENFORCE_NOT_NULL(
        update_stmt.As<ir::Store>(),
        ::common::errors::InvalidArgument(
            "The top-level statement in schedule_block must be a store."));

    auto* store_node = update_stmt.As<ir::Store>();
    ir::Expr rvalue = GetRightOperand(store_node->value);
    PADDLE_ENFORCE_NOT_NULL(rvalue.As<ir::Load>(),
                            ::common::errors::InvalidArgument(
                                "The rvalue of reduce is not a load."));

    std::string func_name = hlir::pe::GridReduceExternalFuncName(
        store_node->value, store_node->tensor->type());
    ir::Tensor rf_tensor = rvalue.As<ir::Load>()->tensor.as_tensor_ref();
    store_node->value =
        lang::CallExtern(func_name, {rf_tensor, num_spatial_threads});
  }

  void Visit(const ir::_LoweredFunc_* expr, ir::Expr* op) override {
    is_after_grid_reduce_ = false;
    func_arg_buffer_names_.clear();
    for (auto& arg : expr->args) {
      if (arg.is_buffer()) {
        func_arg_buffer_names_.insert(arg.buffer_arg()->name);
      }
    }

    IRMutator::Visit(expr, op);
    if (!is_after_grid_reduce_) {
      return;
    }

    ir::_LoweredFunc_* func_node = op->As<ir::_LoweredFunc_>();
    ConvertHeapBuffersToFuncArgs(func_node);
    InsertTempSpaceToFuncArgs(func_node, semaphore_buffer_, true);
    func_node->temp_bufs.push_back(is_done_tensor_->buffer);
  }

  void Visit(const ir::ScheduleBlockRealize* expr, ir::Expr* op) override {
    const ir::ScheduleBlock* schedule_block =
        expr->schedule_block.As<ir::ScheduleBlock>();

    if (schedule_block->name.substr(0, 4) == "root") {
      IRMutator::Visit(expr, op);
      return;
    }

    if (!IsGridReduce(expr)) {
      if (is_after_grid_reduce_) {
        *op = WrapInLastBlockDone(op);
      }
      return;
    }

    PADDLE_ENFORCE_EQ(
        is_after_grid_reduce_,
        false,
        ::common::errors::PreconditionNotMet(
            "Currently supports only one reduce in a fusion group."));
    is_after_grid_reduce_ = true;

    ir::Expr num_spatial_threads = GetThreadBindedSpatialLoopExtend(expr);
    ReplaceByGridReduceExternCall(schedule_block, num_spatial_threads);

    ir::Expr num_spatial_blocks = GetBlockBindedSpatialLoopExtend(expr);
    ir::Expr semaphore_update = CreateSemaphoreUpdateStmt({num_spatial_blocks});
    cur_parent_block_stmts_.push_back(semaphore_update);
    *op = WrapInLastBlockDone(op);
  }

  void Visit(const ir::For* expr, ir::Expr* op) override {
    cur_loops_.push_back(expr);
    IRMutator::Visit(expr, op);
    cur_loops_.pop_back();
  }

  void Visit(const ir::Block* block, ir::Expr* op) override {
    // We override the Block visitor to facilitate statement insertion.
    std::vector<ir::Expr> old_parent_block_stmts;
    old_parent_block_stmts.swap(cur_parent_block_stmts_);
    auto* node = op->As<ir::Block>();
    for (auto& stmt : node->stmts) {
      IRMutator::Visit(&stmt, &stmt);
      cur_parent_block_stmts_.push_back(stmt);
    }
    node->stmts = std::move(cur_parent_block_stmts_);
    cur_parent_block_stmts_ = std::move(old_parent_block_stmts);
  }

  void Visit(ir::Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  std::vector<const ir::For*> cur_loops_;
  std::vector<ir::Expr> cur_parent_block_stmts_;
  std::unordered_set<std::string> func_arg_buffer_names_;
  ir::Tensor is_done_tensor_;
  ir::Buffer semaphore_buffer_;
  bool is_after_grid_reduce_{false};
};

}  // namespace

void ReplaceCrossBlockReduction(Expr* e) { CrossBlockReductionReplacer()(e); }

}  // namespace optim
}  // namespace cinn
