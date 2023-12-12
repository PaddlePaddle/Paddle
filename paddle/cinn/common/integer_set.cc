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

#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace common {

ir::Expr SymbolicExprLimit::positive_inf =
    ir::Expr(ir::Var("positive_infinity"));
ir::Expr SymbolicExprLimit::negative_inf =
    ir::Expr(ir::Var("negative_infinity"));

std::optional<bool> SymbolicExprAnalyzer::Prove(
    const ir::Expr& condition) const {
  if (condition.As<ir::EQ>()) {
    return ProveEQ(condition.As<ir::EQ>()->a(), condition.As<ir::EQ>()->b());
  }
  if (condition.As<ir::NE>()) {
    return ProveNE(condition.As<ir::NE>()->a(), condition.As<ir::NE>()->b());
  }
  if (condition.As<ir::GE>()) {
    return ProveGE(condition.As<ir::GE>()->a(), condition.As<ir::GE>()->b());
  }
  if (condition.As<ir::LE>()) {
    return ProveLE(condition.As<ir::LE>()->a(), condition.As<ir::LE>()->b());
  }
  if (condition.As<ir::GT>()) {
    return ProveGT(condition.As<ir::GT>()->a(), condition.As<ir::GT>()->b());
  }
  if (condition.As<ir::LT>()) {
    return ProveLT(condition.As<ir::LT>()->a(), condition.As<ir::LT>()->b());
  }
  return std::nullopt;
}

std::optional<bool> SymbolicExprAnalyzer::ProveEQ(const ir::Expr& lhs,
                                                  const ir::Expr& rhs) const {
  if (lhs == rhs) {
    return true;
  }
  ir::Expr diff = AutoSimplify(ir::Sub::Make(lhs, rhs), var_intervals_);
  if (diff.is_constant()) {
    return diff.get_constant() == 0;
  }
  std::optional<bool> prove_gt = ProveGT(lhs, rhs);
  if (prove_gt.has_value() && prove_gt.value()) {
    return false;
  }
  std::optional<bool> prove_lt = ProveLT(lhs, rhs);
  if (prove_lt.has_value() && prove_lt.value()) {
    return false;
  }
  return std::nullopt;
}

std::optional<bool> SymbolicExprAnalyzer::ProveNE(const ir::Expr& lhs,
                                                  const ir::Expr& rhs) const {
  if (lhs == rhs) {
    return false;
  }
  ir::Expr diff = AutoSimplify(ir::Sub::Make(lhs, rhs), var_intervals_);
  if (diff.is_constant()) {
    return diff.get_constant() != 0;
  }
  std::optional<bool> prove_gt = ProveGT(lhs, rhs);
  if (prove_gt.has_value() && prove_gt.value()) {
    return true;
  }
  std::optional<bool> prove_lt = ProveLT(lhs, rhs);
  if (prove_lt.has_value() && prove_lt.value()) {
    return true;
  }
  return std::nullopt;
}

std::optional<bool> SymbolicExprAnalyzer::ProveGE(const ir::Expr& lhs,
                                                  const ir::Expr& rhs) const {
  if (lhs == rhs) {
    return true;
  }
  if (lhs == SymbolicExprLimit::positive_inf ||
      rhs == SymbolicExprLimit::negative_inf) {
    return true;
  }
  if (rhs == SymbolicExprLimit::positive_inf ||
      lhs == SymbolicExprLimit::negative_inf) {
    return false;
  }
  ir::Expr diff = AutoSimplify(ir::Sub::Make(lhs, rhs), var_intervals_);
  VLOG(6) << "diff of " << ir::Sub::Make(lhs, rhs) << " = " << diff;
  if (diff.is_constant() && diff.get_constant() >= 0) {
    return true;
  }
  if (diff.is_constant() && diff.get_constant() < 0) {
    return false;
  }
  ir::Expr diff_lower_bound = LowerBound(diff);
  VLOG(6) << "lower bound of " << diff << " = " << diff_lower_bound;
  if (diff_lower_bound.is_constant() && diff_lower_bound.get_constant() >= 0) {
    return true;
  }
  ir::Expr diff_upper_bound = UpperBound(diff);
  VLOG(6) << "upper bound of " << diff << " = " << diff_upper_bound;
  if (diff_upper_bound.is_constant() && diff_upper_bound.get_constant() < 0) {
    return false;
  }
  return std::nullopt;
}

std::optional<bool> SymbolicExprAnalyzer::ProveLE(const ir::Expr& lhs,
                                                  const ir::Expr& rhs) const {
  return ProveGE(rhs, lhs);
}

std::optional<bool> SymbolicExprAnalyzer::ProveGT(const ir::Expr& lhs,
                                                  const ir::Expr& rhs) const {
  if (lhs == rhs) {
    return false;
  }
  if (lhs == SymbolicExprLimit::positive_inf ||
      rhs == SymbolicExprLimit::negative_inf) {
    return true;
  }
  if (rhs == SymbolicExprLimit::positive_inf ||
      lhs == SymbolicExprLimit::negative_inf) {
    return false;
  }
  ir::Expr diff = AutoSimplify(ir::Sub::Make(lhs, rhs), var_intervals_);
  VLOG(6) << "diff of " << ir::Sub::Make(lhs, rhs) << " = " << diff;
  if (diff.is_constant() && diff.get_constant() > 0) {
    return true;
  }
  if (diff.is_constant() && diff.get_constant() <= 0) {
    return false;
  }
  ir::Expr diff_lower_bound = LowerBound(diff);
  VLOG(6) << "lower bound of " << diff << " = " << diff_lower_bound;
  if (diff_lower_bound.is_constant() && diff_lower_bound.get_constant() > 0) {
    return true;
  }
  ir::Expr diff_upper_bound = UpperBound(diff);
  VLOG(6) << "upper bound of " << diff << " = " << diff_upper_bound;
  if (diff_upper_bound.is_constant() && diff_upper_bound.get_constant() <= 0) {
    return false;
  }
  return std::nullopt;
}

std::optional<bool> SymbolicExprAnalyzer::ProveLT(const ir::Expr& lhs,
                                                  const ir::Expr& rhs) const {
  return ProveGT(rhs, lhs);
}

class BoundReplacer : public ir::IRMutator<> {
 public:
  explicit BoundReplacer(const cas_intervals_t& var_intervals,
                         bool is_lower_bound)
      : var_intervals_(var_intervals), sign_(is_lower_bound) {}

  void operator()(ir::Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* var, ir::Expr* op) override {
    ir::Expr lower_bound = SymbolicExprLimit::negative_inf;
    ir::Expr upper_bound = SymbolicExprLimit::positive_inf;
    if (var_intervals_.count(var->name) != 0) {
      const CasInterval& interval = var_intervals_.at(var->name);
      lower_bound =
          interval.e_l.defined() ? interval.e_l : ir::Expr(interval.l);
      upper_bound =
          interval.e_r.defined() ? interval.e_r : ir::Expr(interval.r);
    }
    if (sign_) {
      *op = ir::ir_utils::IRCopy(lower_bound);
    } else {
      *op = ir::ir_utils::IRCopy(upper_bound);
    }
  }

  void Visit(const ir::Add* expr, ir::Expr* op) override {
    ir::Add* node = op->As<ir::Add>();
    IRMutator::Visit(&node->a(), &node->a());
    IRMutator::Visit(&node->b(), &node->b());
  }

  void Visit(const ir::Mul* expr, ir::Expr* op) override {
    ir::Mul* node = op->As<ir::Mul>();
    if (node->b().is_constant() && node->b().get_constant() < 0) {
      sign_ = !sign_;
    }
    IRMutator::Visit(&node->a(), &node->a());
    if (node->b().is_constant() && node->b().get_constant() < 0) {
      sign_ = !sign_;
    }
    if (node->a().is_constant() && node->a().get_constant() < 0) {
      sign_ = !sign_;
    }
    IRMutator::Visit(&node->b(), &node->b());
    if (node->a().is_constant() && node->a().get_constant() < 0) {
      sign_ = !sign_;
    }
  }

  void Visit(const ir::Sub* expr, ir::Expr* op) override {
    ir::Sub* node = op->As<ir::Sub>();
    IRMutator::Visit(&node->a(), &node->a());
    sign_ = !sign_;
    IRMutator::Visit(&node->b(), &node->b());
    sign_ = !sign_;
  }

  void Visit(const ir::Div* expr, ir::Expr* op) override {
    ir::Div* node = op->As<ir::Div>();
    if (node->b().is_constant() && node->b().get_constant() < 0) {
      sign_ = !sign_;
    }
    IRMutator::Visit(&node->a(), &node->a());
    if (node->b().is_constant() && node->b().get_constant() < 0) {
      sign_ = !sign_;
    }
    sign_ = !sign_;
    IRMutator::Visit(&node->b(), &node->b());
    sign_ = !sign_;
  }

  void Visit(const ir::Mod* expr, ir::Expr* op) override {
    ir::Mod* node = op->As<ir::Mod>();
    if (sign_) {
      *op = ir::Expr(0);
    } else {
      IRMutator::Visit(&node->b(), &node->b());
      *op = node->b() - ir::Expr(1);
    }
  }

 private:
  const cas_intervals_t& var_intervals_;
  // Determine replacing with upper or lower bound,
  // True means lower bound and False means upper bound.
  bool sign_;
};

ir::Expr SymbolicExprAnalyzer::LowerBound(const ir::Expr& expr) const {
  BoundReplacer bound_replacer(var_intervals_, true);
  ir::Expr bound = ir::ir_utils::IRCopy(expr);
  bound_replacer(&bound);
  return AutoSimplify(bound);
}

ir::Expr SymbolicExprAnalyzer::UpperBound(const ir::Expr& expr) const {
  BoundReplacer bound_replacer(var_intervals_, false);
  ir::Expr bound = ir::ir_utils::IRCopy(expr);
  bound_replacer(&bound);
  return AutoSimplify(bound);
}

std::optional<bool> ProveEQ(const SingleIntervalIntSet& lhs,
                            const SingleIntervalIntSet& rhs) {
  cas_intervals_t merged_var_intervals = MergeVarIntervals(lhs, rhs);
  SymbolicExprAnalyzer analyzer(merged_var_intervals);
  std::optional<bool> prove_min_eq = analyzer.ProveEQ(lhs.min_, rhs.min_);
  std::optional<bool> prove_max_eq = analyzer.ProveEQ(lhs.max_, rhs.max_);
  if (!prove_min_eq.has_value() || !prove_max_eq.has_value()) {
    return std::nullopt;
  } else if (prove_min_eq.value() == true && prove_max_eq.value() == true) {
    return true;
  } else if (prove_min_eq.value() == false || prove_max_eq.value() == false) {
    return false;
  }
  return std::nullopt;
}

std::optional<SingleIntervalIntSet> ProvedUnion(const SingleIntervalIntSet& a,
                                                const SingleIntervalIntSet& b) {
  bool is_a_empty = a.ProveEmpty().value_or(false);
  bool is_a_all = a.ProveAll().value_or(false);
  bool is_b_empty = b.ProveEmpty().value_or(false);
  bool is_b_all = b.ProveAll().value_or(false);
  if (is_a_empty || is_b_all) {
    return b;
  }
  if (is_b_empty || is_a_all) {
    return a;
  }

  // May be relaxed when (a.max < b.min - 1) or (b.max < a.min - 1)
  cas_intervals_t merged_var_intervals = MergeVarIntervals(a, b);
  SymbolicExprAnalyzer analyzer(merged_var_intervals);
  ir::Expr min = SymbolicExprLimit::positive_inf;
  ir::Expr max = SymbolicExprLimit::negative_inf;

  std::optional<bool> prove_a_min_le_b_min = analyzer.ProveLE(a.Min(), b.Min());
  if (!prove_a_min_le_b_min.has_value()) {
    return std::nullopt;
  } else if (prove_a_min_le_b_min.value() == true) {
    min = a.Min();
  } else if (prove_a_min_le_b_min.value() == false) {
    min = b.Min();
  }

  std::optional<bool> prove_a_max_ge_b_max = analyzer.ProveGE(a.Max(), b.Max());
  if (!prove_a_max_ge_b_max.has_value()) {
    return std::nullopt;
  } else if (prove_a_max_ge_b_max.value() == true) {
    max = a.Max();
  } else if (prove_a_max_ge_b_max.value() == false) {
    max = b.Max();
  }

  return SingleIntervalIntSet(min, max, std::move(merged_var_intervals));
}

std::optional<SingleIntervalIntSet> ProvedIntersect(
    const SingleIntervalIntSet& a, const SingleIntervalIntSet& b) {
  bool is_a_empty = a.ProveEmpty().value_or(false);
  bool is_a_all = a.ProveAll().value_or(false);
  bool is_b_empty = b.ProveEmpty().value_or(false);
  bool is_b_all = b.ProveAll().value_or(false);
  if (is_a_all || is_b_empty) {
    return b;
  }
  if (is_b_all || is_a_empty) {
    return a;
  }
  cas_intervals_t merged_var_intervals = MergeVarIntervals(a, b);
  SymbolicExprAnalyzer analyzer(merged_var_intervals);
  ir::Expr min = SymbolicExprLimit::positive_inf;
  ir::Expr max = SymbolicExprLimit::negative_inf;

  std::optional<bool> prove_a_max_lt_b_min_sub1 =
      analyzer.ProveLT(a.Max(), b.Min() - ir::Expr(1));
  std::optional<bool> prove_b_max_lt_a_min_sub1 =
      analyzer.ProveLT(b.Max(), a.Min() - ir::Expr(1));
  if (prove_a_max_lt_b_min_sub1.has_value() &&
          prove_a_max_lt_b_min_sub1.value() ||
      prove_b_max_lt_a_min_sub1.has_value() &&
          prove_b_max_lt_a_min_sub1.value()) {
    return SingleIntervalIntSet(min, max, std::move(merged_var_intervals));
  }

  std::optional<bool> prove_a_min_ge_b_min = analyzer.ProveGE(a.Min(), b.Min());
  if (!prove_a_min_ge_b_min.has_value()) {
    return std::nullopt;
  } else if (prove_a_min_ge_b_min.value() == true) {
    min = a.Min();
  } else if (prove_a_min_ge_b_min.value() == false) {
    min = b.Min();
  }

  std::optional<bool> prove_a_max_le_b_max = analyzer.ProveLE(a.Max(), b.Max());
  if (!prove_a_max_le_b_max.has_value()) {
    return std::nullopt;
  } else if (prove_a_max_le_b_max.value() == true) {
    max = a.Max();
  } else if (prove_a_max_le_b_max.value() == false) {
    max = b.Max();
  }

  return SingleIntervalIntSet(min, max, std::move(merged_var_intervals));
}

cas_intervals_t MergeVarIntervals(const SingleIntervalIntSet& a,
                                  const SingleIntervalIntSet& b) {
  cas_intervals_t merged = a.var_intervals_;
  merged.insert(b.var_intervals_.begin(), b.var_intervals_.end());
  return merged;
}

SingleIntervalIntSet::SingleIntervalIntSet(const ir::Expr& min,
                                           const ir::Expr& max,
                                           cas_intervals_t var_intervals)
    : min_(min), max_(max), var_intervals_(var_intervals) {
  if (var_intervals_.empty()) {
    auto insert_interval_func = [&](const ir::Expr* x) {
      if (x->as_var()) {
        ir::Expr lower_bound = x->as_var()->lower_bound.defined()
                                   ? x->as_var()->lower_bound
                                   : SymbolicExprLimit::negative_inf;
        ir::Expr upper_bound = x->as_var()->upper_bound.defined()
                                   ? x->as_var()->upper_bound
                                   : SymbolicExprLimit::positive_inf;
        var_intervals_.insert(
            {x->as_var()->name, CasInterval(lower_bound, upper_bound)});
      }
      return false;
    };
    ir::ir_utils::CollectIRNodes(min_, insert_interval_func);
    ir::ir_utils::CollectIRNodes(max_, insert_interval_func);
  }
}

std::optional<bool> SingleIntervalIntSet::ProveEmpty() const {
  if (min_ == SymbolicExprLimit::positive_inf ||
      max_ == SymbolicExprLimit::negative_inf) {
    return true;
  }
  SymbolicExprAnalyzer analyzer(var_intervals_);
  return analyzer.ProveGT(min_, max_);
}

std::optional<bool> SingleIntervalIntSet::ProveAll() const {
  return min_ == SymbolicExprLimit::negative_inf &&
         max_ == SymbolicExprLimit::positive_inf;
}

std::optional<bool> SingleIntervalIntSet::ProvePoint() const {
  SymbolicExprAnalyzer analyzer(var_intervals_);
  return analyzer.ProveEQ(min_, max_);
}

std::optional<bool> SingleIntervalIntSet::ProveSubSet(
    const SingleIntervalIntSet& other) const {
  cas_intervals_t merged_var_intervals = MergeVarIntervals(*this, other);
  SymbolicExprAnalyzer analyzer(merged_var_intervals);
  std::optional<bool> prove_min_ge = analyzer.ProveGE(min_, other.Min());
  std::optional<bool> prove_max_le = analyzer.ProveLE(max_, other.Max());
  if (!prove_min_ge.has_value() || !prove_max_le.has_value()) {
    return std::nullopt;
  } else if (prove_min_ge.value() && prove_max_le.value()) {
    return true;
  } else {
    return false;
  }
  return std::nullopt;
}

std::optional<bool> SingleIntervalIntSet::ProveSuperSet(
    const SingleIntervalIntSet& other) const {
  cas_intervals_t merged_var_intervals = MergeVarIntervals(*this, other);
  SymbolicExprAnalyzer analyzer(merged_var_intervals);
  std::optional<bool> prove_min_le = analyzer.ProveLE(min_, other.Min());
  std::optional<bool> prove_max_ge = analyzer.ProveGE(max_, other.Max());
  if (!prove_min_le.has_value() || !prove_max_ge.has_value()) {
    return std::nullopt;
  } else if (prove_min_le.value() && prove_max_ge.value()) {
    return true;
  } else {
    return false;
  }
  return std::nullopt;
}

}  // namespace common
}  // namespace cinn
