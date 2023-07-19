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

#include "paddle/cinn/optim/cast_bool_to_int8.h"

#include <glog/logging.h>

#include "paddle/cinn/ir/utils/ir_mutator.h"

namespace cinn::optim {

namespace {

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    CHECK(node);
    auto value = node->value;
    if (op->type().is_bool() && op->value->type().is_bool()) {
      value = ir::Cast::Make(Int(8), value);
      *expr = ir::Store::Make(node->tensor, value, node->indices);
    }
  }
};

}  // namespace

void CastBoolToInt8(Expr* e, Target target) {
  if (target.arch == Target::Arch::X86) {
    Mutator mutator;
    mutator.Visit(e, e);
  }
}
}  // namespace cinn::optim
