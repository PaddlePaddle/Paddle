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

#include "paddle/cinn/common/iter_simplify.h"
#include "paddle/cinn/common/const_fold.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_utils.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
namespace cinn {
namespace common {

/*! \brief Override VisitExpr for iter expr type processing */
void IterMapToExprNormalizer::Visit(const Expr* expr, Expr* op) {
  if (auto op_ = op->As<ir::IterSplit>()) {
    *op = ConvertIterSplit(op_);
  } else if (auto op_ = op->As<ir::IterSum>()) {
    *op = ConvertIterSum(op_);
  } else {
    IRMutator::Visit(expr, op);
  }
}

Expr IterMapToExprNormalizer::ConvertIterSum(ir::IterSum* expr) {
  Expr res(0);

  for (auto&& arg : expr->args) {
    auto split = arg.As<ir::IterSplit>();
    res = res + ConvertIterSplit(split);
  }
  res = IsZero(expr->base) ? res : res + expr->base;
  return res;
}

Expr IterMapToExprNormalizer::ConvertIterSplit(ir::IterSplit* expr) {
  // quick branch
  if (IsZero(expr->scale)) return Expr(0);

  Expr source;
  ir::IterMark* mark = expr->source.As<ir::IterMark>();
  if (auto opt = mark->source.As<ir::_Var_>()) {
    source = opt;
  } else if (auto opt = mark->source.As<ir::IterSum>()) {
    source = ConvertIterSum(opt);
  } else {
    VLOG(4) << "unsupported iter expr type";
    Visit(&(mark->source), &(mark->source));
    source = mark->source;
  }
  Expr res;
  if (analyzer_.ProveEQ(expr->extent, mark->extent) &&
      IsOne(expr->lower_factor)) {
    res = source;
  } else if (analyzer_.ProveEQ(mark->extent,
                               expr->lower_factor * expr->extent)) {
    if (IsOne(expr->extent) && !IsOne(mark->extent)) {
      res = ir::Zero(expr->extent.type());
    }
    res = source / expr->lower_factor * expr->scale;
  } else {
    res = (source % (expr->lower_factor * expr->extent)) / expr->lower_factor *
          expr->scale;
  }
  return IsOne(expr->scale) ? res : res * expr->scale;
}

void IterMapRewriter::Visit(const ir::_Var_* op, Expr* expr) {
  auto it = var_map_.find(op->name);
  if (it != var_map_.end()) *expr = it->second;
}

void IterMapRewriter::Visit(const ir::Add* op, Expr* expr) {
  auto a = op->a();
  auto b = op->b();

  Visit(&a);
  Visit(&b);

  if (auto const_res = cinn::common::TryConstFold<ir::Add>(a, b)) {
    *expr = const_res.value();
    return;
  }
  if (!IsIterExpr(a, b)) {
    return;
  }

  Expr ret = ir::ir_utils::IRCopy(ToIterSum(a));

  ir::IterSum* ret_sum = ret.As<ir::IterSum>();

  if (auto b_sum = b.As<ir::IterSum>()) {
    AddToLhs(ret_sum, *b_sum, 1);
  } else if (auto b_split = b.As<ir::IterSplit>()) {
    AddToLhs(ret_sum, *b_split, 1);
  } else {
    ret_sum->base = ret_sum->base + b;
  }
  *expr = ret;
}

void IterMapRewriter::Visit(const ir::Sub* op, Expr* expr) {
  auto a = op->a();
  auto b = op->b();

  Visit(&a);
  Visit(&b);

  if (auto const_res = cinn::common::TryConstFold<ir::Sub>(a, b)) {
    *expr = const_res.value();
    return;
  }
  if (!IsIterExpr(a, b)) return;

  Expr ret = ToIterSum(a);
  ir::IterSum* ret_sum = ret.As<ir::IterSum>();

  if (auto b_sum = b.As<ir::IterSum>()) {
    AddToLhs(ret_sum, *b_sum, -1);
  } else if (auto* b_split = b.As<ir::IterSplit>()) {
    AddToLhs(ret_sum, *b_split, -1);
  } else {
    ret_sum->base = ret_sum->base - b;
  }

  *expr = ret;
}

void IterMapRewriter::Visit(const ir::Mul* op, Expr* expr) {
  auto a = op->a();
  auto b = op->b();

  Visit(&a);
  Visit(&b);

  if (auto const_res = cinn::common::TryConstFold<ir::Mul>(a, b)) {
    *expr = const_res.value();
    return;
  }

  if (!IsIterExpr(a, b)) return;

  if ((a.As<ir::IterSum>() || a.As<ir::IterSplit>()) &&
      (b.As<ir::IterSum>() || b.As<ir::IterSplit>())) {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "product of iter and iter is not supported"));
  }

  if (!a.As<ir::IterSum>() && !a.As<ir::IterSplit>()) {
    std::swap(a, b);
  }

  auto ret = ir::ir_utils::IRCopy(a);

  if (auto a_sum = ret.As<ir::IterSum>()) {
    MulToLhs(a_sum, b);

  } else if (auto a_split = ret.As<ir::IterSplit>()) {
    a_split->scale = a_split->scale * b;
  }

  *expr = ret;
}

Expr IterMapRewriter::ToIterSum(const Expr& expr) {
  if (expr.As<ir::IterSum>()) {
    return expr;
  } else if (auto split = expr.As<ir::IterSplit>()) {
    auto split_expr = ir::IterSplit::Make(
        split->source, split->lower_factor, split->extent, split->scale);
    return ir::IterSum::Make({split_expr}, ir::Zero(expr.type()));
  } else {
    return ir::IterSum::Make({}, expr);
  }
}

void IterMapRewriter::AddToLhs(ir::IterSum* lhs,
                               const ir::IterSplit& rhs,
                               int sign) {
  auto rhs_expr = ir::ir_utils::IRCopy(Expr(const_cast<ir::IterSplit*>(&rhs)));
  for (auto&& lvalue : lhs->args) {
    if (lvalue == rhs_expr) {
      auto lsplit = lvalue.As<ir::IterSplit>();
      if (sign > 0) {
        lsplit->scale = lsplit->scale + rhs.scale;
      } else {
        lsplit->scale = lsplit->scale - rhs.scale;
      }
      return;
    }
  }

  if (sign > 0) {
    lhs->args.push_back(rhs_expr);
  } else {
    rhs_expr.As<ir::IterSplit>()->scale =
        ir::Zero(rhs.scale.type()) - rhs.scale;
    lhs->args.push_back(rhs_expr);
  }
}

void IterMapRewriter::AddToLhs(ir::IterSum* lhs,
                               const ir::IterSum& rhs,
                               int sign) {
  for (auto&& arg : rhs.args) {
    auto rhs = arg.As<ir::IterSplit>();
    AddToLhs(lhs, *rhs, sign);
  }
  if (sign > 0) {
    lhs->base = lhs->base + rhs.base;
  } else {
    lhs->base = lhs->base - rhs.base;
  }
}

void IterMapRewriter::MulToLhs(ir::IterSum* lhs, const Expr& rhs) {
  for (auto&& lvalue : lhs->args) {
    auto lsplit = lvalue.As<ir::IterSplit>();
    lsplit->scale = lsplit->scale * rhs;
  }
  lhs->base = IsZero(lhs->base) ? lhs->base : lhs->base * rhs;
}

}  // namespace common
}  // namespace cinn
