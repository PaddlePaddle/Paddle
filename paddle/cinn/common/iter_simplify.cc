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

ir::IndexExpr IterMapToExprNormalizer::ConvertIterSum(ir::IterSum* expr) {
  ir::IndexExpr res(0);

  for (auto&& arg : expr->args) {
    auto split = arg.As<ir::IterSplit>();
    res = res + ConvertIterSplit(split);
  }

  res = res + expr->base;
  return res;
}

ir::IndexExpr IterMapToExprNormalizer::ConvertIterSplit(ir::IterSplit* expr) {
  // quick branch
  if (IsZero(expr->scale)) return ir::IndexExpr(0);
  ir::IndexExpr source;
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
  if (ProveEQ(expr->extent, mark->extent, analyzer_) &&
      IsOne(expr->lower_factor)) {
    return source * expr->scale;
  } else if (ProveLE(
                 mark->extent, expr->lower_factor * expr->extent, analyzer_)) {
    if (IsOne(expr->extent) && !IsOne(mark->extent)) {
      return ir::Zero(expr->extent.type());
    }

    return source / expr->lower_factor * expr->scale;
  } else {
    return (source % (expr->lower_factor * expr->extent)) / expr->lower_factor *
           expr->scale;
  }
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
    ret_sum->base = ret_sum->base + b.as_index();
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
    ret_sum->base = ret_sum->base - b.as_index();
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
        "Product of iter and iter is not supported"));
    return;
  }

  if (!a.As<ir::IterSum>() && !a.As<ir::IterSplit>()) {
    std::swap(a, b);
  }

  auto ret = ir::ir_utils::IRCopy(a);

  if (auto a_sum = ret.As<ir::IterSum>()) {
    MulToLhs(a_sum, b);

  } else if (auto a_split = ret.As<ir::IterSplit>()) {
    a_split->scale = a_split->scale * b.as_index();
  }

  *expr = ret;
}

void IterMapRewriter::Visit(const ir::Div* op, Expr* expr) {
  auto a = op->a();
  auto b = op->b();

  Visit(&a);
  Visit(&b);

  if (auto const_res = cinn::common::TryConstFold<ir::Div>(a, b)) {
    *expr = const_res.value();
    return;
  }

  if (!IsIterExpr(a, b)) return;

  if ((b.As<ir::IterSum>() || b.As<ir::IterSplit>())) {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "Division of iter and iter is not supported"));
    return;
  }

  auto ret = ir::ir_utils::IRCopy(a);

  auto preprocessed = PreprocessDividend(ret);
  auto preprocessed_sum = preprocessed.As<ir::IterSum>();

  ret = SplitDivConst(preprocessed_sum->args[0], preprocessed_sum->base, b);

  *expr = ret;
}

void IterMapRewriter::Visit(const ir::Mod* op, Expr* expr) {
  auto a = op->a();
  auto b = op->b();

  Visit(&a);
  Visit(&b);

  if (auto const_res = cinn::common::TryConstFold<ir::Mod>(a, b)) {
    *expr = const_res.value();
    return;
  }

  if (!IsIterExpr(a, b)) return;

  if ((b.As<ir::IterSum>() || b.As<ir::IterSplit>())) {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "Mod of iter and iter is not supported"));
    return;
  }

  auto ret = ir::ir_utils::IRCopy(a);

  auto preprocessed = PreprocessDividend(ret);
  auto preprocessed_sum = preprocessed.As<ir::IterSum>();

  ret = SplitModConst(preprocessed_sum->args[0], preprocessed_sum->base, b);

  *expr = ret;
}

Expr IterMapRewriter::PreprocessDividend(const Expr& dividend) {
  if (dividend.As<ir::IterSplit>()) {
    return ir::IterSum::Make({dividend}, ir::Zero(dividend.type()));
  } else if (auto sum = dividend.As<ir::IterSum>()) {
    if (sum->args.size() == 1) {
      return dividend;
    }
    // TODO(liuruyan): number of split in sum is greater then 1, Do `tryFuse` in
    // latter.
    auto fused = dividend;
    return fused;
  } else {
    PADDLE_THROW(
        ::common::errors::InvalidArgument("Expect dividend is IterExpr."));
    return Expr();
  }
}

ir::IndexExpr IterMapRewriter::SplitDivConst(ir::IndexExpr lhs_expr,
                                             ir::IndexExpr base,
                                             ir::IndexExpr rhs) {
  // (lhs_expr + base) // rhs
  if (IsOne(rhs)) {
    if (IsZero(base)) return lhs_expr;
    return ir::IterSum::Make({lhs_expr}, base);
  }

  auto lhs = lhs_expr.As<ir::IterSplit>();
  if (!IsOne(lhs->scale)) {
    if (ProveDivisible(lhs->scale, rhs, analyzer_) && IsZero(base)) {
      lhs->scale = lhs->scale / rhs;
      return lhs;
    } else if (ProveDivisible(lhs->scale, rhs, analyzer_) &&
               ProveDivisible(base, rhs, analyzer_)) {
      lhs->scale = lhs->scale / rhs;
      return ir::IterSum::Make({lhs}, base / rhs);
    } else if (ProveDivisible(rhs, lhs->scale, analyzer_) && IsZero(base)) {
      rhs = rhs / lhs->scale;
      lhs->scale = ir::One(rhs.type());
    } else if (ProveDivisible(rhs, lhs->scale, analyzer_) &&
               ProveDivisible(base, lhs->scale, analyzer_)) {
      base = base / lhs->scale;
      rhs = rhs / lhs->scale;
      lhs->scale = ir::One(rhs.type());
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "IterExpr scale must be divisible by rhs"));
      return ir::IndexExpr();
    }
  }

  // TODO(liuruyan): Padding dividend to divisor later. assuming dividend canbe
  // divided by divisor now.

  ir::IndexExpr new_split;
  if (!ProveDivisible(base, rhs, analyzer_)) {
    // padding base to divisor later. Treat the whole expr as IterMark now.
    return ir::IterSum::Make(
        {ir::IterSplit::Make(
            ir::IterMark::Make(ir::IterSum::Make({ir::IndexExpr(lhs)}, base),
                               lhs->extent + base),
            rhs,
            (lhs->extent + base + rhs - 1) / rhs,
            ir::One(rhs.type()))},
        ir::Zero(rhs.type()));
  }

  if (ProveDivisible(lhs->extent, rhs, analyzer_)) {
    new_split = ir::IterSplit::Make(
        lhs->source, lhs->lower_factor * rhs, lhs->extent / rhs, lhs->scale);
  } else if (IsOne(lhs->lower_factor) &&
             ProveEQ(lhs->extent,
                     lhs->source.As<ir::IterMark>()->extent,
                     analyzer_)) {
    new_split = ir::IterSplit::Make(
        lhs->source, rhs, (lhs->extent + rhs - 1) / rhs, lhs->scale);
  } else {
    new_split = ir::IterSplit::Make(ir::IterMark::Make(lhs, lhs->extent),
                                    rhs,
                                    (lhs->extent + rhs - 1) / rhs,
                                    ir::One(rhs.type()));
  }

  return ir::IterSum::Make({new_split}, base / rhs);
}

ir::IndexExpr IterMapRewriter::SplitModConst(ir::IndexExpr lhs_expr,
                                             ir::IndexExpr base,
                                             ir::IndexExpr rhs) {
  // (lhs_expr + base) % rhs
  if (IsOne(rhs)) {
    return ir::Zero(lhs_expr.type());
  }

  auto lhs = lhs_expr.As<ir::IterSplit>();
  if (!IsOne(lhs->scale)) {
    if (ProveDivisible(lhs->scale, rhs, analyzer_) && IsZero(base)) {
      return ir::Zero(lhs_expr.type());
    } else if (ProveDivisible(lhs->scale, rhs, analyzer_) &&
               ProveDivisible(base, rhs, analyzer_)) {
      return ir::Zero(lhs_expr.type());
    } else if (ProveDivisible(rhs, lhs->scale, analyzer_) && IsZero(base)) {
      rhs = rhs / lhs->scale;
    } else if (ProveDivisible(rhs, lhs->scale, analyzer_) &&
               ProveDivisible(base, lhs->scale, analyzer_)) {
      base = base / lhs->scale;
      rhs = rhs / lhs->scale;
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "IterExpr scale must be divisible by rhs"));
      return ir::IndexExpr();
    }
  }

  if (!ProveDivisible(base, rhs, analyzer_)) {
    auto lhs_s1 = ir::IterSplit::Make(
        lhs->source, lhs->lower_factor, lhs->extent, ir::One(lhs_expr.type()));
    // padding base to divisor later. Treat the whole expr as IterMark now.
    return ir::IterSplit::Make(
        ir::IterMark::Make(ir::IterSum::Make({lhs_s1}, base),
                           lhs->extent + base),
        ir::One(rhs.type()),
        rhs,
        lhs->scale);
  }
  // TODO(liuruyan): Padding dividend to divisor later. assuming dividend canbe
  // divided by divisor now.

  return ir::IterSplit::Make(lhs->source, lhs->lower_factor, rhs, lhs->scale);
}

ir::IndexExpr IterMapRewriter::ToIterSum(const Expr& expr) {
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
  auto rhs_expr =
      ir::ir_utils::IRCopy(Expr(const_cast<ir::IterSplit*>(&rhs))).as_index();
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
        ir::Zero(rhs.scale.type()).as_index() - rhs.scale;
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

void IterMapRewriter::MulToLhs(ir::IterSum* lhs, const ir::IndexExpr& rhs) {
  for (auto&& lvalue : lhs->args) {
    auto lsplit = lvalue.As<ir::IterSplit>();
    lsplit->scale = lsplit->scale * rhs;
  }
  lhs->base = lhs->base * rhs;
}

}  // namespace common
}  // namespace cinn
