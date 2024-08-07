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

#include "paddle/cinn/optim/if_fusion.h"

#include <stack>
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/optim/ir_simplify.h"

#define VisitImpl(_TYPE)                                 \
  void Visit(const ir::_TYPE *op, Expr *expr) override { \
    last_op = Expr(const_cast<ir::_TYPE *>(op));         \
    ir::IRMutator<>::Visit(op, expr);                    \
  }

namespace cinn {
namespace optim {

namespace {

struct IfFusionMutator : public ir::IRMutator<Expr *> {
  void operator()(Expr *expr) { Visit(expr, expr); }

 private:
  void Visit(const ir::IfThenElse *op, Expr *expr) override {
    // the implementation of ifFusion
    // compare the last condition with current condition
    // judge whether last_op is nullptr
    if (!last_op.get()) {
      last_op = Expr(const_cast<ir::IfThenElse *>(op));
      return;
    }

    // judge whether last_op is IfThenElse
    ir::IfThenElse *lop = last_op.As<ir::IfThenElse>();
    if (!lop) {
      last_op = Expr(const_cast<ir::IfThenElse *>(op));
      return;
    }

    // judge whether condition is same
    bool is_need_fuse = ir::ir_utils::IRCompare(op->condition, lop->condition);
    if (is_need_fuse) {
      // do fusion (cop.true_case <-> lop.true_case)
      Fuse(op->true_case, lop->true_case);

      // support for recursive true case merge
      Expr tmp = last_op;
      Visit(&lop->true_case, &lop->true_case);
      last_op = tmp;

      if (op->false_case.defined() && lop->false_case.defined()) {
        Fuse(op->false_case, lop->false_case);
        // support for recusive false case merge
        tmp = last_op;
        Visit(&lop->false_case, &lop->false_case);
        last_op = tmp;
      }

      // Remove the op which refers to current ir::IfThenElse block,
      // because this block is merged with previous ir::IfThenElse block,
      // so blank now.
      // push the elements position which will be deleted after visit current
      // block.
      RecordIndexForErase(Expr(const_cast<ir::IfThenElse *>(op)), cur_block);
    }

    if (!is_need_fuse) {
      last_op = Expr(const_cast<ir::IfThenElse *>(op));
    }
  }

  void Visit(const ir::Block *op, Expr *expr) override {
    int element_num_before_visit = erase_elements_ind.size();
    ir::Block *last_block = (cur_block);
    cur_block = const_cast<ir::Block *>(op);
    ir::IRMutator<>::Visit(op, expr);
    cur_block = last_block;

    EraseBlankElements(const_cast<ir::Block *>(op), element_num_before_visit);
  }

  // Recode for the sequent Erasure
  void RecordIndexForErase(Expr op, ir::Block *cur_block) {
    for (int i = 0; i < cur_block->stmts.size(); i++) {
      if (ir::ir_utils::IRCompare(cur_block->stmts[i], op)) {
        erase_elements_ind.push(i);
        return;
      }
    }
  }

  // Erase the blank block
  void EraseBlankElements(ir::Block *op, int stack_upper_bound) {
    while (erase_elements_ind.size() > stack_upper_bound) {
      int erase_pos = erase_elements_ind.top();
      erase_elements_ind.pop();
      op->stmts.erase(op->stmts.begin() + erase_pos);
    }
  }

  VisitImpl(Expr);
  VisitImpl(ScheduleBlock);
  VisitImpl(For);
  VisitImpl(IntImm);
  VisitImpl(UIntImm);
  VisitImpl(FloatImm);
  VisitImpl(StringImm);
  VisitImpl(Cast);
  VisitImpl(PolyFor);
  VisitImpl(Select);
  VisitImpl(Call);
  VisitImpl(_Module_);
  VisitImpl(_Var_);
  VisitImpl(Load);
  VisitImpl(Store);
  VisitImpl(Alloc);
  VisitImpl(Free);
  VisitImpl(_Buffer_);
  VisitImpl(_Tensor_);
  VisitImpl(_LoweredFunc_);
  VisitImpl(Let);
  VisitImpl(Reduce);
  VisitImpl(Ramp);
  VisitImpl(Broadcast);
  VisitImpl(FracOp);
  VisitImpl(Product);
  VisitImpl(Sum);
  VisitImpl(PrimitiveNode);
  VisitImpl(IntrinsicOp);
  VisitImpl(_BufferRange_);
  VisitImpl(_Dim_);

  void Fuse(Expr ne, Expr oe) {
    // fuse old expr with new expr, merge the stmts in them.
    ir::Block *neb = ne.As<ir::Block>();
    ir::Block *oeb = oe.As<ir::Block>();

#ifdef __cpp_lib_containers_range
    oeb->stmts.append_range(neb->stmts);
#else
    oeb->stmts.insert(oeb->stmts.end(), neb->stmts.cbegin(), neb->stmts.cend());
#endif

    neb->stmts.clear();
  }

  std::stack<int> erase_elements_ind;

  // record the condition of it if last block is if-block, nullptr otherwise.
  Expr last_op = Expr(nullptr);

  ir::Block *cur_block;
};  // IfFusionMutator
}  // namespace

void IfFusion(Expr *expr) { IfFusionMutator()(expr); }
}  // namespace optim
}  // namespace cinn
