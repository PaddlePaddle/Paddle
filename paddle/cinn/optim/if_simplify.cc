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

#include "paddle/cinn/optim/if_simplify.h"

#include "paddle/cinn/ir/utils/ir_mutator.h"

namespace cinn::optim {

namespace {

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::IfThenElse* op, Expr* expr) {
    auto* condition_int = op->condition.As<ir::IntImm>();
    auto* condition_uint = op->condition.As<ir::UIntImm>();
    int64_t value;
    if (condition_int || condition_uint) {
      if (condition_int) {
        value = condition_int->value;
      } else {
        value = condition_uint->value;
      }
      if (value) {
        *expr = op->true_case;
      } else {
        if (op->false_case.defined()) {
          *expr = op->false_case;
        } else {
          // null condition
          *expr = ir::Block::Make({});
        }
      }
    }
  }
};

}  // namespace

void IfSimplify(Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
}

}  // namespace cinn::optim
