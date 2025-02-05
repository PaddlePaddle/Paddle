// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/replace_mod_to_max.h"

#include <unordered_map>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"

namespace cinn {
namespace optim {

/**
 * Replace Mod to possible max value.
 * a % b -> min(b - 1, a)
 * either b - 1 or a is the possible max value of the mod expression.
 */
class ReplaceModToMaxMutator : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Mod* op, ir::Expr* expr) override {
    ir::Mod* node = expr->As<ir::Mod>();
    Expr base = ir::Sub::Make(node->operand(1), Expr(1));
    Expr min_expr = ir::Min::Make(node->operand(0), base);
    *expr = cinn::optim::ArithSimplify(min_expr);
    ir::IRMutator<>::Visit(expr, expr);
  }
};

void ReplaceModToMax(ir::Expr* expr) {
  ReplaceModToMaxMutator mutator;
  mutator(expr);
}

}  // namespace optim
}  // namespace cinn
