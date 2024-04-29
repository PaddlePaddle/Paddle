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

#include "paddle/cinn/optim/eliminate_broadcast_in_forloop.h"

#include <tuple>
#include <vector>

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/utils/ir_replace.h"

namespace cinn {
namespace optim {

namespace detail {

struct EliminateBroadcastInForloop : public ir::IRMutator<Expr*> {
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Store* op, Expr* expr) {
    // TODO(Superjom) Support single one level of forloop.
    if (forloop_stack.size() < 2) return;

    auto* node = expr->As<ir::Store>();

    auto broadcasts = ir::ir_utils::CollectIRNodes(
        node->value,
        [&](const Expr* expr) { return expr->As<ir::Broadcast>(); });
    std::vector<Expr> let_exprs;

    Var tmp;
    Expr let_expr;

    Var cur_level_loop_var =
        forloop_stack.back()->As<ir::For>()
            ? forloop_stack.back()->As<ir::For>()->loop_var
            : forloop_stack.back()->As<ir::PolyFor>()->iterator;
    for (Expr broadcast : broadcasts) {
      if (ContainsLoopVar(broadcast, cur_level_loop_var)) continue;
      VLOG(4) << "eliminating " << broadcast;
      std::tie(let_expr, tmp) = CreateTmpLet(broadcast);
      let_exprs.push_back(let_expr);

      cinn::ir::ir_utils::IrReplaceVarBroadcast(expr, broadcast, tmp);
    }

    // insert the let expressions to the outer forloop.

    Expr* outer_forloop = forloop_stack[forloop_stack.size() - 2];

    auto& outer_forloop_body = outer_forloop->As<ir::For>()
                                   ? outer_forloop->As<ir::For>()->body
                                   : outer_forloop->As<ir::PolyFor>()->body;

    auto* outer_forloop_body_block = outer_forloop_body.As<ir::Block>();
    if (outer_forloop_body_block) {
      outer_forloop_body_block->stmts.insert(
          std::begin(outer_forloop_body_block->stmts),
          let_exprs.begin(),
          let_exprs.end());

    } else {
      let_exprs.push_back(outer_forloop_body);
      outer_forloop_body = ir::Block::Make(let_exprs);
    }
  }

  bool ContainsLoopVar(Expr expr, Var loop_var) {
    return !ir::ir_utils::CollectIRNodes(expr, [&](const Expr* e) -> bool {
              return e->As<ir::_Var_>() &&
                     e->As<ir::_Var_>()->name == loop_var->name;
            }).empty();
  }

  std::tuple<Expr, Var> CreateTmpLet(Expr body) {
    Var tmp(Context::Global().NewName("tmp"), body.type());

    Expr let_expr = ir::Let::Make(tmp, body);

    return std::make_tuple(let_expr, tmp);
  }

  void Visit(const ir::For* op, Expr* expr) {
    forloop_stack.push_back(expr);
    ir::IRMutator<>::Visit(op, expr);
    forloop_stack.pop_back();
  }

  void Visit(const ir::PolyFor* op, Expr* expr) {
    forloop_stack.push_back(expr);
    ir::IRMutator<>::Visit(op, expr);
    forloop_stack.pop_back();
  }

  std::vector<Expr*> forloop_stack;
};

}  // namespace detail

void EliminateBroadcastInForloop(Expr* expr) {
  detail::EliminateBroadcastInForloop mutator;
  mutator(expr);
}

}  // namespace optim
}  // namespace cinn
