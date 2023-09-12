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

#include "paddle/cinn/optim/collect_undefined_vars.h"

#include <set>

#include "paddle/cinn/ir/utils/ir_mutator.h"

namespace cinn::optim {

namespace {
struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;
  std::vector<std::string> undefined_vars;
  std::set<std::string> defined_vars;
  std::set<std::string> used_vars;

  void CollectVarDef(const std::string& var) {
    CHECK(!defined_vars.count(var))
        << "var " << var << " has been defined, please check";
    CHECK(!used_vars.count(var))
        << "var " << var << " is wrongly used before definition";
    defined_vars.insert(var);
  }

  void ClearVar(const std::string& var) {
    defined_vars.erase(var);
    used_vars.erase(var);
  }

  void CollectVarUse(const std::string& var) {
    used_vars.insert(var);
    if (defined_vars.count(var) == 0) {
      undefined_vars.push_back(var);
    }
  }

  void Visit(const ir::Let* op, Expr* expr) final {
    Expr symbol = op->symbol;
    auto var = symbol.as_var_ref();
    CHECK(var.defined());
    CollectVarDef(var->name);
    auto* node = expr->As<ir::Let>();
    Visit(&node->body, &node->body);
  }

  void Visit(const ir::For* op, Expr* expr) final {
    CollectVarDef(op->loop_var->name);
    auto* node = expr->As<ir::For>();
    Visit(&node->min, &node->min);
    Visit(&node->extent, &node->extent);
    Visit(&node->body, &node->body);
    ClearVar(op->loop_var->name);
  }

  void Visit(const ir::Load* op, Expr* expr) final {
    auto tensor = op->tensor.as_tensor_ref();
    CollectVarUse(tensor->name);
    auto* node = expr->As<ir::Load>();
    for (auto& idx : node->indices) Visit(&idx, &idx);
  }

  void Visit(const ir::Store* op, Expr* expr) final {
    auto tensor = op->tensor.as_tensor_ref();
    CollectVarUse(tensor->name);
    auto* node = expr->As<ir::Store>();
    for (auto& idx : node->indices) Visit(&idx, &idx);
    Visit(&node->value, &node->value);
  }

  void Visit(const ir::_Var_* op, Expr* expr) final {
    CollectVarUse(op->name);
    auto* node = expr->As<ir::_Var_>();
    if (node->lower_bound.defined()) {
      Visit(&node->lower_bound, &node->lower_bound);
    }
    if (node->upper_bound.defined()) {
      Visit(&node->upper_bound, &node->upper_bound);
    }
  }

  void Visit(const ir::Reduce* op, Expr* expr) final {
    for (auto& axis : op->reduce_axis) {
      CollectVarDef(axis->name);
    }
    auto* node = expr->As<ir::Reduce>();
    if (node->init.defined()) Visit(&node->init, &node->init);
    Visit(&node->body, &node->body);
  }
};
}  // namespace

std::vector<std::string> CollectUndefinedVars(Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
  return mutator.undefined_vars;
}

}  // namespace cinn::optim
