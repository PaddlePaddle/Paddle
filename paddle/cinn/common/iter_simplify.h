// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/common/iter_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"

namespace cinn {
namespace common {

class IterMapToExprNormalizer : public ir::IRMutator<> {
 public:
  explicit IterMapToExprNormalizer(const SymbolicExprAnalyzer& analyzer)
      : analyzer_(analyzer) {}

  void Convert(Expr* expr) { Visit(expr, expr); }

 private:
  void Visit(const Expr* expr, Expr* op) override;

  ir::IndexExpr ConvertIterSum(ir::IterSum* expr);

  ir::IndexExpr ConvertIterSplit(ir::IterSplit* expr);

 private:
  common::SymbolicExprAnalyzer analyzer_;
};

class IterMapRewriter : public ir::IRMutator<> {
 public:
  explicit IterMapRewriter(const std::vector<ir::Var>& input_iters,
                           const SymbolicExprAnalyzer& analyzer)
      : analyzer_(analyzer) {
    for (const auto& iter : input_iters) {
      if (IsOne(iter->upper_bound)) {
        var_map_[iter->name] = ir::IterSum::Make({}, iter->lower_bound);
      } else if (IsZero(iter->lower_bound)) {
        auto tmp =
            ir::IterMark::Make(ir::IndexExpr(iter.ptr()), iter->upper_bound);
        auto mark = tmp.As<ir::IterMark>();
        var_map_[iter->name] = ir::IterSplit::Make(tmp);
        input_marks_.push_back(*mark);
      } else {
        PADDLE_THROW(::common::errors::InvalidArgument(
            "iter should start from 0, but got %d", iter->lower_bound));
      }
    }
  }

  void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Rewrite(Expr* expr) {
    IRMutator::Visit(expr, expr);
    *expr = ToIterSum(*expr);
  }

  void Visit(const ir::_Var_* op, Expr* expr) override;

  void Visit(const ir::Add* op, Expr* expr) override;

  void Visit(const ir::Sub* op, Expr* expr) override;

  void Visit(const ir::Mul* op, Expr* expr) override;

  void Visit(const ir::Div* op, Expr* expr) override;

  void Visit(const ir::Mod* op, Expr* expr) override;

 private:
  static ir::IndexExpr ToIterSum(const Expr& expr);

  static void AddToLhs(ir::IterSum* lhs, const ir::IterSplit& rhs, int sign);

  static void AddToLhs(ir::IterSum* lhs, const ir::IterSum& rhs, int sign);

  static void MulToLhs(ir::IterSum* lhs, const ir::IndexExpr& rhs);

  Expr PreprocessDividend(const Expr& dividend);

  ir::IndexExpr SplitDivConst(ir::IndexExpr lhs,
                              ir::IndexExpr base,
                              ir::IndexExpr rhs);

  ir::IndexExpr SplitModConst(ir::IndexExpr lhs,
                              ir::IndexExpr base,
                              ir::IndexExpr rhs);

  std::unordered_map<std::string, ir::IndexExpr> var_map_;
  std::vector<ir::IterMark> input_marks_;
  common::SymbolicExprAnalyzer analyzer_;
};

}  // namespace common
}  // namespace cinn
