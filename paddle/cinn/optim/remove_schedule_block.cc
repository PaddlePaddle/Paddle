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

#include "paddle/cinn/optim/remove_schedule_block.h"

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

struct ScheduleBlockRemover : public ir::IRMutator<Expr*> {
  void operator()(ir::Expr* expr) { Visit(expr); }

 private:
  void Visit(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override {
    auto* node = expr->As<ir::ScheduleBlockRealize>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument(
            "The expression could not be cast to ir::ScheduleBlockRealize. "
            "Please check the expression type."));
    auto& iter_values = node->iter_values;
    auto* schedule_block = node->schedule_block.As<ir::ScheduleBlock>();
    PADDLE_ENFORCE_NOT_NULL(
        schedule_block,
        ::common::errors::InvalidArgument(
            "The schedule block could not be cast to ir::ScheduleBlock. Please "
            "check the schedule block type."));
    auto& iter_vars = schedule_block->iter_vars;
    Expr body = schedule_block->body;
    PADDLE_ENFORCE_EQ(iter_vars.size(),
                      iter_values.size(),
                      ::common::errors::InvalidArgument(
                          "The size of iter vars and iter values is not equal,"
                          "where iter vars:%d but iter values:%d.",
                          iter_vars.size(),
                          iter_values.size()));
    for (int i = 0; i < iter_vars.size(); i++) {
      optim::ReplaceVarWithExpr(&body, iter_vars[i], iter_values[i]);
    }
    *expr = body;
    IRMutator::Visit(expr, expr);
  }
};

void RemoveScheduleBlock(Expr* e) { ScheduleBlockRemover()(e); }

}  // namespace optim
}  // namespace cinn
