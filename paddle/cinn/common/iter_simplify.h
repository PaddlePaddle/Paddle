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

#pragma once
#include <optional>
#include <unordered_map>
#include <vector>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"

namespace cinn {
namespace common {

class IterMapToExprNormalizer : public ir::IRMutator<> {
 public:
  explicit IterMapToExprNormalizer(SymbolicExprAnalyzer analyzer)
      : analyzer_(analyzer) {}

  void Convert(Expr* expr) { Visit(expr, expr); }

 private:
  void Visit(const Expr* expr, Expr* op) override;

  Expr ConvertIterSum(ir::IterSum* expr);

  Expr ConvertIterSplit(ir::IterSplit* expr);

 private:
  common::SymbolicExprAnalyzer& analyzer_;
};

class IterMapRewriter : public ir::IRMutator<> {
 public:
  explicit IterMapRewriter(const std::vector<ir::Var>& input_iters);

  void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Rewrite(Expr* expr) {
    IRMutator::Visit(expr, expr);
    *expr = ToIterSum(*expr);
  }

  void Visit(const ir::_Var_* op, Expr* expr) override;

  void Visit(const ir::Add* op, Expr* expr) override;

  void Visit(const ir::Sub* op, Expr* expr) override;

  void Visit(const ir::Mul* op, Expr* expr) override;

 private:
  static Expr ToIterSum(const Expr& expr);

  static void AddToLhs(ir::IterSum* lhs, const ir::IterSplit& rhs, int sign);

  static void AddToLhs(ir::IterSum* lhs, const ir::IterSum& rhs, int sign);

  static void MulToLhs(ir::IterSum* lhs, const Expr& rhs);

  std::unordered_map<std::string, Expr> var_map_;
  std::vector<ir::IterMark> input_marks_;
};

}  // namespace common
}  // namespace cinn
