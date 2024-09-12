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

#include "paddle/cinn/optim/rearrange_load_instruction.h"

#include <stack>
#include <vector>
#include "paddle/cinn/adt/map_expr.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/optim/ir_simplify.h"

#define VisitImpl(_TYPE)                                 \
  void Visit(const ir::_TYPE *op, Expr *expr) override { \
    auto old_last_op = last_op;                          \
    auto *ncop = const_cast<ir::_TYPE *>(op);            \
    last_op = Expr(ncop);                                \
    ir::IRMutator<>::Visit(op, expr);                    \
    last_op = old_last_op;                               \
  }

namespace cinn {
namespace optim {

namespace {

struct RearrangeLoadInstructionMutator : public ir::IRMutator<Expr *> {
  RearrangeLoadInstructionMutator() {
    is_inner_store = false;
    last_op = Expr(nullptr);
  }
  void operator()(Expr *expr) { Visit(expr, expr); }

 private:
  std::vector<ir::Store *> local_stores;
  bool is_local_store(ir::Load *op) {
    for (int i = 0; i < local_stores.size(); i++) {
      if (ir::ir_utils::IRCompare(local_stores[i]->tensor, op->tensor)) {
        if (local_stores[i]->indices.size() != op->indices.size()) continue;
        bool all_same = true;
        for (int j = 0; j < op->indices.size(); j++) {
          if (!ir::ir_utils::IRCompare(local_stores[i]->indices[j],
                                       op->indices[j])) {
            all_same = false;
          }
        }
        if (all_same) return true;
      }
    }
    return false;
  }
  void Visit(const ir::Load *op, Expr *expr) override {
    auto load_op = expr->As<ir::Load>();
    if (is_inner_store) {
      if (op->tensor.as_tensor_ref()->buffer.operator->() != nullptr &&
              (op->tensor.as_tensor_ref()->buffer->memory_type ==
                   ir::MemoryType::GPULocal ||
               op->tensor.as_tensor_ref()->buffer->memory_type ==
                   ir::MemoryType::GPUShared) ||
          is_local_store(load_op))
        return;

      auto local_var =
          ir::_Var_::Make(common::UniqName("local_var"), op->type());
      auto let_op = ir::Let::Make(local_var, const_cast<ir::Load *>(op));
      let_list.push_back(let_op);
      last_op->replace(Expr(const_cast<ir::Load *>(op)), local_var);
    }
  }

  void Visit(const ir::Store *op, Expr *expr) override {
    auto store_op = expr->As<ir::Store>();
    auto old_last_op = last_op;
    local_stores.push_back(store_op);
    last_op = Expr(store_op);
    is_inner_store = true;
    ir::IRMutator<>::Visit(op, expr);
    is_inner_store = false;
    last_op = old_last_op;
  }

  void Visit(const ir::Block *op, Expr *expr) override {
    auto old_last_op = last_op;
    last_op = Expr(const_cast<ir::Block *>(op));
    int old_let_size = let_list.size();
    int old_stmts_size = stmts_list.size();

    for (auto &stmt : op->stmts) {
      IRVisitorRequireReImpl<void, Expr *>::Visit(
          &stmt, const_cast<ir::Expr *>(&stmt));
      stmts_list.push_back(stmt);
    }

    if (let_list.size() > old_let_size) {
      replaceBlock(const_cast<ir::Block *>(op), old_let_size, old_stmts_size);
    }
    last_op = old_last_op;
  }

  void Visit(const ir::Expr *op, Expr *expr) override {
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Select *op, Expr *expr) override {}
  void Visit(const ir::Broadcast *op, Expr *expr) override {}

  void replaceBlock(ir::Block *op, int old_let_size, int old_stmts_size) {
    std::vector<Expr> new_stmts;

    for (int i = old_let_size; i < let_list.size(); i++) {
      new_stmts.push_back(let_list[i]);
    }

    for (int i = old_stmts_size; i < stmts_list.size(); i++) {
      new_stmts.push_back(stmts_list[i]);
    }

    while (let_list.size() > old_let_size) {
      let_list.pop_back();
    }

    while (stmts_list.size() > old_stmts_size) {
      stmts_list.pop_back();
    }

    op->stmts = new_stmts;
  }

  std::unordered_map<std::string, ir::Expr> collection_name_map_expr;
  std::vector<Expr> let_list;
  std::vector<Expr> stmts_list;
  bool is_inner_store;
  Expr last_op;

  VisitImpl(ScheduleBlock);
  VisitImpl(For);
  VisitImpl(Cast);
  VisitImpl(PolyFor);
  VisitImpl(Call);
  VisitImpl(_Module_);
  VisitImpl(_Var_);
  VisitImpl(Alloc);
  VisitImpl(Free);
  VisitImpl(_Buffer_);
  VisitImpl(_Tensor_);
  VisitImpl(_LoweredFunc_);
  VisitImpl(Let);
  VisitImpl(Reduce);
  VisitImpl(Ramp);
  VisitImpl(FracOp);
  VisitImpl(Product);
  VisitImpl(Sum);
  VisitImpl(PrimitiveNode);
  VisitImpl(IntrinsicOp);
  VisitImpl(_BufferRange_);
  VisitImpl(_Dim_);

  VisitImpl(Add);
  VisitImpl(Sub);
  VisitImpl(Mul);
  VisitImpl(Div);
  VisitImpl(Mod);
  VisitImpl(EQ);
  VisitImpl(NE);
  VisitImpl(LT);
  VisitImpl(LE);
  VisitImpl(GT);
  VisitImpl(GE);
  VisitImpl(And);
  VisitImpl(Or);
  VisitImpl(Not);
  VisitImpl(Min);
  VisitImpl(Max);
  VisitImpl(Minus)
};
}  // namespace

void RearrangeLoadInstruction(Expr *expr) {
  RearrangeLoadInstructionMutator collector;
  collector(expr);
}

}  // namespace optim
}  // namespace cinn
