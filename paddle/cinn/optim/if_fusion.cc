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

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/optim/ir_simplify.h"

namespace cinn {
namespace optim {

namespace {
struct IfFusionMutator : public ir::IRMutator<Expr *> {
  void operator()(Expr *expr) {
    ir::IRMutator<>::Visit(expr, expr);

    // simplify from the root of tree, for removing the blank node.
    Simplify(expr);
  }

 private:
  void Visit(const ir::IfThenElse *op, Expr *expr) override {
    // the implementation of ifFusion
    // compare the last condition with current condition
    // judge whether last_op is nullptr
    if (!last_op.get()) return;

    // judge whether last_op is IfThenElse
    const ir::IfThenElse *lop = last_op.As<const ir::IfThenElse>();
    if (!lop) return;

    // ir::IfThenElse *cop = op->As<ir::IfThenElse>();
    //  judge whether condition is same
    bool isNeedFuse = ir::ir_utils::IRCompare(op->condition, lop->condition);
    if (isNeedFuse) {
      // do fusion (cop.true_case <-> lop.true_case)
      Fuse(op->true_case, lop->true_case);
      // do fusion (cop.false_case <-> lop.false_case)
      Fuse(op->false_case, lop->false_case);
    }
    // else {
    //   last_op = op;
    // }

    if (!isNeedFuse) {
      last_op = Expr(const_cast<ir::IfThenElse *>(op));
    }
  }

  void Visit(const Expr *op, Expr *expr) override {
    ir::IRMutator<>::Visit(op, expr);
    // last_op = static_cast<const Expr *>(op);
    last_op = Expr(const_cast<ir::Expr *>(op));
  }

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

  // record the condition of it if last block is if-block, nullptr otherwise.
  Expr last_op = Expr(nullptr);
};  // IfFusionMutator
}  // namespace

void IfFusion(Expr *expr) { IfFusionMutator()(expr); }
}  // namespace optim
}  // namespace cinn
