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
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"

namespace cinn {
namespace common {

int64_t GetLargestMutiplyPart(const Expr& expr) {
  switch (expr.node_type()) {
    case cinn::ir::IrNodeTy::_Var_:
      return 1;
    case cinn::ir::IrNodeTy::Div: {
      auto binExpr = expr.As<ir::Div>();
      auto rhs = binExpr->b();
      if (rhs.type().is_index_type()) {
        int64_t lhsDiv = GetLargestMutiplyPart(binExpr->a());
        int64_t rhsDiv = GetLargestMutiplyPart(binExpr->b());
        if (lhsDiv % rhsDiv == 0) return std::abs(lhsDiv / rhsDiv);
      }
      return 1;
    }
    case cinn::ir::IrNodeTy::IntImm: {
      auto int_imm = expr.As<ir::IntImm>();
      return std::abs(int_imm->value);
    }
    case cinn::ir::IrNodeTy::Mul: {
      auto binExpr = expr.As<ir::Div>();
      return GetLargestMutiplyPart(binExpr->a()) *
             GetLargestMutiplyPart(binExpr->b());
    }
    case cinn::ir::IrNodeTy::Sub:
      [[fallthrough]];
    case cinn::ir::IrNodeTy::Add:
      [[fallthrough]];
    case cinn::ir::IrNodeTy::Mod: {
      return std::gcd(GetLargestMutiplyPart(expr.ptr()->operand(0)),
                      GetLargestMutiplyPart(expr.ptr()->operand(1)));
    }
  }
  PADDLE_THROW(::common::errors::Unimplemented("Unsupported type of expr: %s",
                                               expr.type()));
}

bool IsIterExpr(const Expr& a, const Expr& b) {
  return a.As<ir::IterSplit>() || a.As<ir::IterSum>() ||
         b.As<ir::IterSplit>() || b.As<ir::IterSum>();
}

bool IsOne(const Expr& expr) {
  if (expr.is_constant() && expr.get_constant() == 1) {
    return true;
  }
  return false;
}
bool IsZero(const Expr& expr) {
  if (expr.is_constant() && expr.get_constant() == 0) {
    return true;
  }
  return false;
}
bool ProveDivisible(const Expr& lhs,
                    const Expr& rhs,
                    const common::SymbolicExprAnalyzer& analyzer) {
  if (auto rhs_imm = rhs.As<ir::IntImm>()) {
    return GetLargestMutiplyPart(lhs) % rhs_imm->value == 0;
  } else if (rhs.is_var()) {
    return analyzer.ProveDivisible(lhs, rhs).value_or(false);
  } else {
    return false;
  }
}

bool ProveEQ(const Expr& lhs,
             const Expr& rhs,
             const common::SymbolicExprAnalyzer& analyzer) {
  if (lhs == rhs) return true;
  return analyzer.ProveEQ(lhs, rhs).value_or(false);
}

bool ProveLE(const Expr& lhs,
             const Expr& rhs,
             const common::SymbolicExprAnalyzer& analyzer) {
  if (lhs == rhs) return true;
  return analyzer.ProveLE(lhs, rhs).value_or(false);
}

template <typename TNode, typename FLeaf>
inline void UnpackReduction(const ir::IndexExpr& value, FLeaf fleaf) {
  if (const TNode* node = value.As<TNode>()) {
    UnpackReduction<TNode, FLeaf>(node->a(), fleaf);
    UnpackReduction<TNode, FLeaf>(node->b(), fleaf);
  } else {
    fleaf(value);
  }
}

// TODO(liuruyan): canby simplify into IndexExpr multiply.
inline ir::IndexExpr MulAndNormalize(const ir::IndexExpr& lhs,
                                     const ir::IndexExpr& rhs) {
  int64_t cscale = 1;
  ir::IndexExpr res = ir::One(lhs.type());
  auto fcollect = [&](ir::IndexExpr val) {
    if (const auto* intimm = val.As<ir::IntImm>()) {
      cscale *= intimm->value;
    } else {
      res = res * val;
    }
  };
  UnpackReduction<ir::Mul>(lhs, fcollect);
  UnpackReduction<ir::Mul>(rhs, fcollect);
  if (cscale != 1) {
    res = res * ir::IndexExpr(make_shared<ir::IntImm>(res.type(), cscale));
  }
  return res;
}
}  // namespace common
}  // namespace cinn
