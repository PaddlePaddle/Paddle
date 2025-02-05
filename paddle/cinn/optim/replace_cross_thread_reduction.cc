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
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/pass/pass_manager.h"

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
struct CrossThreadReductionReplacer {
  void operator()(ir::LoweredFunc fn) { Visit(fn.As<ir::_LoweredFunc_>()); }

 private:
  bool CanReplace(const ir::stmt::Schedule block) {
    if (block->name().substr(0, 4) == "root") {
      return false;
    }

    const std::vector<ir::Expr>& iter_values = block->iter_values();
    const std::vector<ir::Var>& iter_vars = block->iter_vars();

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

    auto IsThreadBindOnReduceAxis = [&](const ir::stmt::For& for_node) {
      return reduce_var_names.count(for_node->loop_var()->name) > 0 &&
             for_node->is_gpu_thread_binded();
    };

    std::vector<int> thread_binded_reduce_loop_indices;
    bool is_thread_binded_inner_loop = false;
    for (int i = 0; i < cur_loops_.size(); ++i) {
      bool is_thread_bind_on_reduce = IsThreadBindOnReduceAxis(cur_loops_[i]);
      if (is_thread_bind_on_reduce && ir::GetLoopExtent(cur_loops_[i]) == 1) {
        return false;
      }
      if (is_thread_binded_inner_loop || is_thread_bind_on_reduce) {
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
      if (loop->is_gpu_thread_binded()) {
        block_size *= ir::GetLoopExtent(loop);
      }
    }
    return block_size;
  }

  template <typename OpT>
  void ReplaceByContinuousReduceExternCall(ir::stmt::Store store,
                                           bool return_warp) {
    auto* node = store->value().As<OpT>();
    PADDLE_ENFORCE_NOT_NULL(
        node, ::common::errors::InvalidArgument("The node must not be null."));
    auto& operand = node->b();
    std::string reduce_func_name = hlir::pe::CrossThreadReduceExternalFuncName(
        store->value(), operand.template As<ir::Load>()->tensor);
    auto tmp_dtype =
        operand.template As<ir::Load>()->tensor.as_tensor()->type();
    auto tmp_buffer = ir::_Buffer_::Make(
        "shm32_" + hlir::pe::Type2StrForReduce(tmp_dtype) + "_reduce",
        {ir::Expr(32)});
    tmp_buffer->dtype = tmp_dtype;
    tmp_buffer->memory_type = ir::MemoryType::GPUShared;
    shm_buffer_.insert(tmp_buffer);
    store->set_value(lang::CallExtern(
        reduce_func_name, {node->b(), tmp_buffer, ir::Expr(return_warp)}));
  }

  template <typename OpT>
  void ReplaceByDiscreteReduceExternCall(ir::stmt::Store store) {
    auto* node = store->value().As<OpT>();
    PADDLE_ENFORCE_NOT_NULL(
        node, ::common::errors::InvalidArgument("The node must not be null."));
    auto& operand = node->b();
    std::string reduce_func_name = hlir::pe::DiscreteReduceExternalFuncName(
        store->value(), operand.template As<ir::Load>()->tensor);
    auto tmp_dtype =
        operand.template As<ir::Load>()->tensor.as_tensor()->type();
    auto tmp_buffer = ir::_Buffer_::Make(
        "shm32_" + hlir::pe::Type2StrForReduce(tmp_dtype) + "_reduce",
        {ir::Expr(GetBlockSize())});
    tmp_buffer->dtype = tmp_dtype;
    tmp_buffer->memory_type = ir::MemoryType::GPUShared;
    shm_buffer_.insert(tmp_buffer);
    store->set_value(
        lang::CallExtern(reduce_func_name, {node->b(), tmp_buffer}));
  }

  template <typename OpT>
  void ReplaceByReduceExternCall(ir::stmt::Store store,
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

  void Visit(ir::_LoweredFunc_* fn) {
    ir::stmt::Mutate(
        fn->body_block,
        [&](ir::stmt::StmtRef stmt) { PreCall(stmt); },
        [&](ir::stmt::StmtRef stmt) { PostCall(stmt); });
    if (std::find_if(fn->temp_bufs.begin(),
                     fn->temp_bufs.end(),
                     [&](const ir::Buffer& buf) -> bool {
                       for (auto& tmp_buf : shm_buffer_) {
                         if (buf->name == tmp_buf->name) return true;
                       }
                       return false;
                     }) == fn->temp_bufs.end())
      fn->temp_bufs.insert(
          fn->temp_bufs.end(), shm_buffer_.begin(), shm_buffer_.end());
    shm_buffer_.clear();
  }

  void PreCall(ir::stmt::StmtRef stmt) {
    switch (stmt->stmt_type()) {
      case ir::StmtNodeTy::Schedule:
        VisitStmt(stmt.as<ir::stmt::Schedule>());
        break;
      case ir::StmtNodeTy::For:
        cur_loops_.push_back(stmt.as<ir::stmt::For>());
        break;
      default:
        break;
    }
  }

  void PostCall(ir::stmt::StmtRef stmt) {
    switch (stmt->stmt_type()) {
      case ir::StmtNodeTy::For:
        cur_loops_.pop_back();
        break;
      default:
        break;
    }
  }

  void VisitStmt(ir::stmt::Schedule stmt) {
    if (!CanReplace(stmt)) {
      return;
    }
    ir::stmt::BlockRef original_update_body = stmt->body();

    ir::stmt::Store original_update_stmt;
    PADDLE_ENFORCE_EQ(original_update_body->stmts().size(),
                      1,
                      ::common::errors::InvalidArgument(
                          "The size of statements is incorrect."
                          "Expected size is 1, but receive %d.",
                          original_update_body->stmts().size()));
    PADDLE_ENFORCE_EQ(original_update_body->stmts()[0].isa<ir::stmt::Store>(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The stmt in schedule's body should be store "
                          "statement, but get %s.",
                          original_update_body->stmts()[0]->stmt_type()));
    original_update_stmt =
        original_update_body->stmts()[0].as<ir::stmt::Store>();

    switch (original_update_stmt->value()->node_type()) {
      case cinn::ir::IrNodeTy::Add:
        ReplaceByReduceExternCall<ir::Add>(original_update_stmt,
                                           stmt->reduce_method());
        break;
      case cinn::ir::IrNodeTy::Mul:
        ReplaceByReduceExternCall<ir::Mul>(original_update_stmt,
                                           stmt->reduce_method());
        break;
      case cinn::ir::IrNodeTy::Max:
        ReplaceByReduceExternCall<ir::Max>(original_update_stmt,
                                           stmt->reduce_method());
        break;
      case cinn::ir::IrNodeTy::Min:
        ReplaceByReduceExternCall<ir::Min>(original_update_stmt,
                                           stmt->reduce_method());
        break;
      case cinn::ir::IrNodeTy::And:
        ReplaceByReduceExternCall<ir::And>(original_update_stmt,
                                           stmt->reduce_method());
        break;
      case cinn::ir::IrNodeTy::Or:
        ReplaceByReduceExternCall<ir::Or>(original_update_stmt,
                                          stmt->reduce_method());
        break;
      default:
        PADDLE_THROW(::common::errors::InvalidArgument(
            "The node type is not supported in cross thread reduction."));
    }
  }

 private:
  std::vector<ir::stmt::For> cur_loops_;
};

}  // namespace

void ReplaceCrossThreadReduction(ir::LoweredFunc fn) {
  FuncPassManager manager;
  manager.AddPass(std::make_unique<ReplaceCrossThreadReductionPass>());
  manager.Run(fn);
}

LogicalResult ReplaceCrossThreadReductionPass::Run(ir::LoweredFunc func) {
  CrossThreadReductionReplacer replacer;
  replacer(func);
  return LogicalResult::success();
}

}  // namespace optim
}  // namespace cinn
