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

#include "paddle/cinn/optim/replace_var_with_expr.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/replace_const_param_to_integer.h"

namespace cinn {
namespace optim {

struct ReplaceVarWithExprMutator : public ir::IRMutator<> {
  ReplaceVarWithExprMutator(const Var& var,
                            const Expr& expr,
                            const std::string& tensor_name)
      : var_(var), expr_(expr), tensor_name_(tensor_name) {}

  void operator()(Expr* expr) {
    if (tensor_name_.empty()) visit_all_ = true;
    IRMutator::Visit(expr, expr);
  }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (expr->name == var_->name && (do_replace_ || visit_all_)) {
      auto copied = ir::ir_utils::IRCopy(expr_);
      *op = copied;
    }
  }

  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    ir::IRMutator<>::Visit(&node->min, &node->min);
    ir::IRMutator<>::Visit(&node->extent, &node->extent);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    if (node->loop_var->name == var_->name && expr_.As<ir::_Var_>() &&
        visit_all_) {
      node->loop_var = expr_.As<ir::_Var_>();
    }
  }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    ir::IRMutator<>::Visit(&node->init, &node->init);
    ir::IRMutator<>::Visit(&node->condition, &node->condition);
    ir::IRMutator<>::Visit(&node->inc, &node->inc);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    if (node->iterator->name == var_->name && expr_.As<ir::_Var_>() &&
        visit_all_) {
      node->iterator = expr_.As<ir::_Var_>();
    }
  }

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    auto* tensor = node->tensor.as_tensor();

    if (tensor->name == tensor_name_) {
      do_replace_ = true;
    } else {
      do_replace_ = false;
    }
    for (auto& index : node->indices) {
      ir::IRMutator<>::Visit(&index, &index);
    }
    do_replace_ = false;
    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
    ir::IRMutator<>::Visit(&node->value, &node->value);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    auto* node = op->As<ir::Load>();
    auto* tensor = node->tensor.as_tensor();
    if (tensor->name == tensor_name_) {
      do_replace_ = true;
    } else {
      do_replace_ = false;
    }
    for (auto& idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
    do_replace_ = false;
    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
  }

 private:
  bool do_replace_{false};
  bool visit_all_{false};
  const Var& var_;
  const Expr& expr_;
  const std::string& tensor_name_;
};

void ReplaceVarWithExpr(Expr* source,
                        const Var& var,
                        const Expr& expr,
                        const std::string& tensor_name) {
  ReplaceVarWithExprMutator mutator(var, expr, tensor_name);
  mutator(source);
}

struct CollectTensorIndexMutator : public ir::IRMutator<> {
  explicit CollectTensorIndexMutator(const std::string& tensor_name)
      : tensor_name_(tensor_name) {}

  std::vector<std::vector<Expr>> operator()(Expr* expr) {
    IRMutator::Visit(expr, expr);
    return res;
  }

 private:
  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    ir::IRMutator<>::Visit(&node->body, &node->body);
  }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    ir::IRMutator<>::Visit(&node->body, &node->body);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    auto* node = op->As<ir::Load>();
    auto* tensor = node->tensor.as_tensor();
    if (tensor->name == tensor_name_) {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      res.push_back(node->indices);
    } else {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      for (auto& idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
    }
  }

 private:
  std::vector<std::vector<Expr>> res;
  const std::string& tensor_name_;
};

std::vector<std::vector<Expr>> CollectTensorIndex(
    Expr* source, const std::string& tensor_name) {
  CollectTensorIndexMutator mutator(tensor_name);
  std::vector<std::vector<Expr>> result = mutator(source);
  for (auto& i : result) {
    for (auto& j : i) {
      j = common::AutoSimplify(j);
    }
  }
  return result;
}

}  // namespace optim
}  // namespace cinn
