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

#include "paddle/cinn/optim/replace_const_param_to_integer.h"

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/poly/ast_gen.h"
#include "paddle/cinn/utils/string.h"

namespace cinn::optim {

namespace {

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::_Var_* op, Expr* expr) override {
    if (utils::StartsWith(op->name, poly::kIslParamConstPrefix)) {
      std::string value = op->name.substr(strlen(poly::kIslParamConstPrefix));
      *expr = Expr(std::stoi(value));
    }
  }
};

}  // namespace

void ReplaceConstParamToInteger(Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
}

}  // namespace cinn::optim
