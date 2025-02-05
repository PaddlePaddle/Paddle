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
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
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
  ir::IndexExpr source;
  ir::IterMark* mark = expr->source.As<ir::IterMark>();
  if (auto opt = mark->source.As<ir::_Var_>()) {
    if (IsOne(mark->extent)) return ir::IndexExpr(0);
    source = ir::IndexExpr(opt);
  } else if (auto opt = mark->source.As<ir::IterSum>()) {
    source = ConvertIterSum(opt);
  } else {
    VLOG(4) << "unsupported iter expr type";
    Visit(&(mark->source), &(mark->source));
    source = mark->source;
  }

  // quick branch
  if (IsZero(expr->scale) || IsOne(expr->extent))
    return ir::Zero(expr->extent.type());

  if (analyzer_.ProveEQ(expr->extent, mark->extent).value_or(false) &&
      IsOne(expr->lower_factor)) {
    return source * expr->scale;
  } else if (analyzer_.ProveLE(mark->extent, expr->lower_factor * expr->extent)
                 .value_or(false)) {
    if (IsOne(expr->extent)) {
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

  Expr ret = ir::ir_utils::IRCopy(ToIterSum(a));
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
    auto opt_fused = TryFuse(dividend);
    if (!opt_fused) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Dividend can't be written as a single fused IterSum"));
      return Expr();
    }
    return opt_fused.value();
  } else {
    PADDLE_THROW(
        ::common::errors::InvalidArgument("Expect dividend is IterExpr."));
    return Expr();
  }
}

Expr IterMapRewriter::SplitDivConst(Expr lhs_expr,
                                    ir::IndexExpr base,
                                    ir::IndexExpr rhs) {
  // (lhs_expr + base) // rhs
  if (IsOne(rhs)) {
    if (IsZero(base)) return lhs_expr;
    return ir::IterSum::Make({lhs_expr}, base);
  }

  auto lhs = lhs_expr.As<ir::IterSplit>();
  if (!IsOne(lhs->scale)) {
    if (ProveDivisible(lhs->scale, rhs) && IsZero(base)) {
      lhs->scale = lhs->scale / rhs;
      return lhs;
    } else if (ProveDivisible(lhs->scale, rhs) && ProveDivisible(base, rhs)) {
      lhs->scale = lhs->scale / rhs;
      return ir::IterSum::Make({lhs}, base / rhs);
    } else if (ProveDivisible(rhs, lhs->scale) && IsZero(base)) {
      rhs = rhs / lhs->scale;
      lhs->scale = ir::One(rhs.type());
    } else if (ProveDivisible(rhs, lhs->scale) &&
               ProveDivisible(base, lhs->scale)) {
      base = base / lhs->scale;
      rhs = rhs / lhs->scale;
      lhs->scale = ir::One(rhs.type());
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "IterExpr scale must be divisible by rhs"));
      return Expr();
    }
  }

  // TODO(liuruyan): Padding dividend to divisor later. assuming dividend canbe
  // divided by divisor now.

  Expr new_split;
  if (!ProveDivisible(base, rhs)) {
    // padding base to divisor later. Treat the whole expr as IterMark now.
    return ir::IterSum::Make(
        {ir::IterSplit::Make(
            ir::IterMark::Make(ir::IterSum::Make({Expr(lhs)}, base),
                               lhs->extent + base),
            rhs,
            (lhs->extent + base + rhs - 1) / rhs,
            ir::One(rhs.type()))},
        ir::Zero(rhs.type()));
  }

  if (ProveDivisible(lhs->extent, rhs)) {
    new_split = ir::IterSplit::Make(
        lhs->source, lhs->lower_factor * rhs, lhs->extent / rhs, lhs->scale);
  } else if (IsOne(lhs->lower_factor) &&
             analyzer_
                 .ProveEQ(lhs->extent, lhs->source.As<ir::IterMark>()->extent)
                 .value_or(false)) {
    new_split = ir::IterSplit::Make(
        lhs->source, rhs, (lhs->extent + rhs - 1) / rhs, lhs->scale);
  } else {
    new_split = ir::IterSplit::Make(ir::IterMark::Make(lhs, lhs->extent),
                                    rhs,
                                    (lhs->extent + rhs - 1) / rhs,
                                    ir::One(rhs.type()));
  }
  return IsZero(base / rhs) ? new_split
                            : ir::IterSum::Make({new_split}, base / rhs);
}

Expr IterMapRewriter::SplitModConst(ir::Expr lhs_expr,
                                    ir::IndexExpr base,
                                    ir::IndexExpr rhs) {
  // (lhs_expr + base) % rhs
  if (IsOne(rhs)) {
    return ir::Zero(lhs_expr.type());
  }

  auto lhs = lhs_expr.As<ir::IterSplit>();
  if (!IsOne(lhs->scale)) {
    if (ProveDivisible(lhs->scale, rhs) && IsZero(base)) {
      return ir::Zero(lhs_expr.type());
    } else if (ProveDivisible(lhs->scale, rhs) && ProveDivisible(base, rhs)) {
      return ir::Zero(lhs_expr.type());
    } else if (ProveDivisible(rhs, lhs->scale) && IsZero(base)) {
      rhs = rhs / lhs->scale;
    } else if (ProveDivisible(rhs, lhs->scale) &&
               ProveDivisible(base, lhs->scale)) {
      base = base / lhs->scale;
      rhs = rhs / lhs->scale;
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "IterExpr scale must be divisible by rhs"));
      return Expr();
    }
  }

  if (!ProveDivisible(base, rhs)) {
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

int32_t IterMapRewriter::FindFirstPossibleUnitExtentIndex(
    const ir::IterSum& expr) {
  for (int32_t i = 0; i < expr.args.size(); ++i) {
    if (IsOne(expr.args[i].As<ir::IterSplit>()->extent)) return i;
  }
  return static_cast<int32_t>(expr.args.size());
}

int32_t IterMapRewriter::FindSplitWithExactScale(
    const ir::IterSum& expr,
    const std::vector<bool>& skip_flag,
    const ir::IndexExpr& expected_scale,
    const Expr& match_source,
    int32_t rbegin,
    int32_t first_possible_unit_extent_pos) {
  if (rbegin == -1) {
    rbegin = static_cast<int32_t>(expr.args.size()) - 1;
  }
  int32_t matched_pos = -1;
  // Use reverse search, as smallest scale usually are near the end.
  for (int32_t j = rbegin; j >= 0; --j) {
    if (skip_flag[j]) continue;
    auto split = expr.args[j].As<ir::IterSplit>();
    if (match_source.defined() && match_source != split->source) continue;
    const ir::IndexExpr& cur_scale = split->scale;
    if (analyzer_.ProveEQ(cur_scale, expected_scale).value_or(false)) {
      if (IsOne(split->extent)) return j;
      // We prefer the unit extent Iter. just search when extent != 1.
      if (matched_pos == -1) {
        matched_pos = j;
      }
      // There is no unit extent in front of first_possible_unit_extent_pos,
      // so just return.
      if (j <= first_possible_unit_extent_pos) return matched_pos;
    }
  }
  return matched_pos;
}

int32_t IterMapRewriter::FindBaseSplit(const ir::IterSum& expr,
                                       const std::vector<bool>& skip_flag,
                                       const Expr& match_source,
                                       int32_t rbegin) {
  if (rbegin == -1) {
    rbegin = static_cast<int>(expr.args.size()) - 1;
  }

  int32_t base_index = -1;
  int64_t min_const_scale = 0;

  // Compare the const scale size, use reverse search, as smallest scale usually
  // are near the end.
  for (int32_t i = rbegin; i >= 0; --i) {
    if (skip_flag[i]) continue;
    auto split = expr.args[i].As<ir::IterSplit>();
    if (match_source.defined() && match_source != split->source) continue;
    if (const auto* op = split->scale.As<ir::IntImm>()) {
      if (base_index == -1 || op->value < min_const_scale) {
        min_const_scale = op->value;
        base_index = i;
      } else if (op->value == min_const_scale) {
        if (IsOne(split->extent) &&
            !IsOne(expr.args[base_index].As<ir::IterSplit>()->extent)) {
          base_index = i;
        }
      }
    }
  }

  // Found! return the base index.
  if (base_index != -1) return base_index;

  // If not found const scale, compare the symbole length in scale.
  int32_t min_reduce_size = 0;
  for (int32_t i = rbegin; i >= 0; --i) {
    if (skip_flag[i]) continue;
    auto split = expr.args[i].As<ir::IterSplit>();
    if (match_source.defined() && match_source != split->source) continue;
    int32_t reduce_size = 0;
    auto fcollect = [&](const ir::IndexExpr&) { ++reduce_size; };
    UnpackReduction<ir::Mul>(split->scale, fcollect);
    if (base_index == -1 || reduce_size < min_reduce_size) {
      min_reduce_size = reduce_size;
      base_index = i;
    }
  }
  return base_index;
}

std::optional<Expr> IterMapRewriter::TryFuse(const Expr& expr) {
  auto iter_sum = expr.As<ir::IterSum>();
  if (!iter_sum) return std::nullopt;
  if (iter_sum->args.size() <= 1) return std::nullopt;

  // Fuse Iter with same source. e.g. i_j_fused / 4 * 4 + i_j_fused % 4
  if (auto opt = TryFuseSameSource(expr)) {
    auto sum = opt.value().As<ir::IterSum>();
    if (sum->args.size() <= 1) {
      return opt.value();
    }
  }

  // Select iter with smallest scale as base iter.
  std::vector<bool> visited(iter_sum->args.size(), false);
  int base_index = FindBaseSplit(*iter_sum, visited, Expr(), -1);
  if (base_index == -1) return std::nullopt;
  ir::IndexExpr base_scale =
      iter_sum->args[base_index].As<ir::IterSplit>()->scale;

  std::vector<Expr> grouped_iters;

  ir::IndexExpr expected_scale = base_scale;
  int first_possible_unit_extent_pos =
      FindFirstPossibleUnitExtentIndex(*iter_sum);

  // Find iter with same scale as expected_scale and update expected_scale.
  // e.g. i * 32 + j * 8 + k * 1, Extent(i, j, k) = 2, 4, 8.
  // first base_index = 2, expected_scale = 1. means select k as base iter.
  // then matched_pos = 2, expected_scale = 8 * 1 = 8. means match k.
  // then matched_pos = 1, expected_scale = 8 * 4 = 32. means match j.
  // finally matched_pos = 0, expected_scale = 32 * 2 = 64. means match i.
  // if match failed, indicates that expr is illegal and cannot be merged.
  for (size_t i = 0; i < iter_sum->args.size(); ++i) {
    ir::IndexExpr matched_scale;
    int matched_pos =
        i == 0 ? base_index
               : FindSplitWithExactScale(*iter_sum,
                                         visited,
                                         expected_scale,
                                         Expr(),
                                         -1,
                                         first_possible_unit_extent_pos);
    // If not found iter with expected scale, search above case:
    // D(i)=2, D(j)=8, Split loop from (j, 0, 8) to (-1, 32)
    // (i * 8 + j) % 16 ==> (i * 8 + j0 * 32 + j1)
    if (matched_pos == -1) {
      matched_pos = FindBaseSplit(*iter_sum, visited, Expr(), -1);
      // // If not found iter with expected scale again, return nullopt.
      if (matched_pos == -1) return std::nullopt;
    }

    matched_scale = expected_scale;
    visited[matched_pos] = true;
    auto arg_copy = ir::ir_utils::IRCopy(iter_sum->args[matched_pos]);
    auto arg = arg_copy.As<ir::IterSplit>();
    arg->scale = arg->scale / base_scale;
    grouped_iters.push_back(arg_copy);

    // Update expected_scale = matched_split->scale * matched_split->extent
    expected_scale =
        iter_sum->args[matched_pos].As<ir::IterSplit>()->extent * matched_scale;
  }
  std::reverse(grouped_iters.begin(), grouped_iters.end());
  Expr grouped_sum =
      ir::IterSum::Make(grouped_iters, ir::Zero(iter_sum->type()));

  // If the iter is already fused, return it directly.
  auto it = sum_fuse_map_.find(grouped_sum);
  if (it != sum_fuse_map_.end()) {
    return ir::IterSum::Make({ir::IterSplit::Make(it->second, base_scale)},
                             iter_sum->base);
  } else {
    // new iter, form a new mark
    auto mark = ir::IterMark::Make(grouped_sum, expected_scale / base_scale);
    sum_fuse_map_[grouped_sum] = mark;
    return ir::IterSum::Make({ir::IterSplit::Make(mark, base_scale)},
                             iter_sum->base);
  }
}

std::optional<Expr> IterMapRewriter::TryFuseSameSource(const Expr& expr) {
  auto iter_sum = expr.As<ir::IterSum>();
  if (!iter_sum) return std::nullopt;
  if (iter_sum->args.size() <= 1) return std::nullopt;

  // Only for IterMark
  std::unordered_map<Expr, int32_t> hit_count;

  bool has_overlap = false;
  // Check if the iterators have overlap, just return nullopt if not.
  for (auto&& split : iter_sum->args) {
    auto mark = split.As<ir::IterSplit>()->source;
    auto it = hit_count.find(mark);
    if (it != hit_count.end()) {
      ++it->second;
      has_overlap = true;
    } else {
      hit_count[mark] = 1;
    }
  }
  if (!has_overlap) return std::nullopt;

  std::vector<bool> visited(iter_sum->args.size(), false);
  // Only for IterSplit
  std::vector<Expr> reverse_flattened_iters;

  int first_possible_unit_extent_pos =
      FindFirstPossibleUnitExtentIndex(*iter_sum);

  // Start eliminating the iterators
  for (int rend = static_cast<int32_t>(iter_sum->args.size()) - 1; rend >= 0;) {
    auto split = iter_sum->args[rend].As<ir::IterSplit>();
    if (visited[rend]) {
      --rend;
      continue;
    }
    if (hit_count.at(split->source) == 1) {
      reverse_flattened_iters.push_back(iter_sum->args[rend]);
      visited[rend] = true;
      --rend;
      continue;
    }
    int matched_index = FindBaseSplit(*iter_sum, visited, split->source, rend);
    visited[matched_index] = true;
    auto split_copy = ir::ir_utils::IRCopy(iter_sum->args[matched_index]);
    auto rhs_iter = split_copy.As<ir::IterSplit>();

    // Eliminate the lhs iterators when meets the following conditions:
    // 1. The lhs has the same source as the rhs.
    // 2. lhs->scale == rhs->extent * rhs->scale.
    // 3. lhs->lower_factor == rhs->lower_factor * rhs->extent.
    while (true) {
      ir::IndexExpr lhs_scale = rhs_iter->extent * rhs_iter->scale;
      matched_index = FindSplitWithExactScale(*iter_sum,
                                              visited,
                                              lhs_scale,
                                              rhs_iter->source,
                                              rend,
                                              first_possible_unit_extent_pos);
      if (matched_index == -1) break;
      auto lhs_iter = iter_sum->args[matched_index].As<ir::IterSplit>();
      ir::IndexExpr lhs_lower_factor =
          rhs_iter->lower_factor * rhs_iter->extent;
      if (!analyzer_.ProveEQ(lhs_iter->lower_factor, lhs_lower_factor)
               .value_or(false))
        break;
      visited[matched_index] = true;

      rhs_iter->extent = lhs_iter->extent * rhs_iter->extent;
    }
    reverse_flattened_iters.push_back(split_copy);
  }
  std::reverse(reverse_flattened_iters.begin(), reverse_flattened_iters.end());
  auto simplified_sum =
      ir::IterSum::Make(reverse_flattened_iters, iter_sum->base);
  return simplified_sum;
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
  auto rhs_expr = ir::ir_utils::IRCopy(Expr(&Reference(&rhs)));
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
    rhs_expr.As<ir::IterSplit>()->scale = -rhs.scale;
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

void IterMapSimplify(std::vector<Expr>& indices,  // NOLINT
                     const std::vector<cinn::ir::Var>& input_iters,
                     const SymbolicExprAnalyzer& analyzer) {
  IterMapRewriter rewriter(input_iters, analyzer);
  IterMapToExprNormalizer converter(analyzer);
  for (auto& value : indices) {
    VLOG(5) << "before rewrite: " << value;
    rewriter.Rewrite(&value);
    VLOG(5) << "after rewrite: " << value;
    converter.Convert(&value);
    VLOG(5) << "after convert: " << value;
  }
}

void SimplifyBlockBinding::Visit(const ir::For* op, Expr* expr) {
  auto for_op = expr->As<ir::For>();
  loop_var_.emplace_back(op->loop_var);
  IRMutator::Visit(for_op, expr);
  loop_var_.pop_back();
}

void SimplifyBlockBinding::Visit(const ir::ScheduleBlockRealize* op,
                                 Expr* expr) {
  auto sch_block_rlz_op = expr->As<ir::ScheduleBlockRealize>();
  if (sch_block_rlz_op->iter_values.empty()) return;
  IterMapSimplify(sch_block_rlz_op->iter_values, loop_var_, analyzer_);
}

}  // namespace common
}  // namespace cinn
