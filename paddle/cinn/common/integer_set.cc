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

bool SymbolicExprAnalyzer::CanProve(const ir::Expr& condition) const {
  if (condition.As<ir::EQ>()) {
    return CanProveEQ(condition.As<ir::EQ>()->a(), condition.As<ir::EQ>()->b());
  }
  if (condition.As<ir::NE>()) {
    return CanProveNE(condition.As<ir::NE>()->a(), condition.As<ir::NE>()->b());
  }
  if (condition.As<ir::GE>()) {
    return CanProveGE(condition.As<ir::GE>()->a(), condition.As<ir::GE>()->b());
  }
  if (condition.As<ir::LE>()) {
    return CanProveLE(condition.As<ir::LE>()->a(), condition.As<ir::LE>()->b());
  }
  if (condition.As<ir::GT>()) {
    return CanProveGT(condition.As<ir::GT>()->a(), condition.As<ir::GT>()->b());
  }
  if (condition.As<ir::LT>()) {
    return CanProveLT(condition.As<ir::LT>()->a(), condition.As<ir::LT>()->b());
  }
  CINN_NOT_IMPLEMENTED;
  return false;
}

bool SymbolicExprAnalyzer::CanProveEQ(const ir::Expr& lhs,
                                      const ir::Expr& rhs) const {
  if (lhs == rhs) {
    return true;
  }
  ir::Expr diff = AutoSimplify(ir::Sub::Make(lhs, rhs), *var_intervals_);
  return (diff.is_constant() && diff.get_constant() == 0);
}

bool SymbolicExprAnalyzer::CanProveNE(const ir::Expr& lhs,
                                      const ir::Expr& rhs) const {
  return CanProveGT(lhs, rhs) || CanProveLT(lhs, rhs);
}

bool SymbolicExprAnalyzer::CanProveGE(const ir::Expr& lhs,
                                      const ir::Expr& rhs) const {
  ir::Expr diff = AutoSimplify(ir::Sub::Make(lhs, rhs), *var_intervals_);
  VLOG(6) << "diff of " << ir::Sub::Make(lhs, rhs) << " = " << diff;
  if (diff.is_constant() && diff.get_constant() >= 0) {
    return true;
  }
  ir::Expr diff_lower_bound = LowerBound(diff);
  VLOG(6) << "lower bound of " << diff << " = " << diff_lower_bound;
  if (diff_lower_bound.is_constant() && diff_lower_bound.get_constant() >= 0) {
    return true;
  }
  return false;
}

bool SymbolicExprAnalyzer::CanProveLE(const ir::Expr& lhs,
                                      const ir::Expr& rhs) const {
  return CanProveGE(rhs, lhs);
}

bool SymbolicExprAnalyzer::CanProveGT(const ir::Expr& lhs,
                                      const ir::Expr& rhs) const {
  ir::Expr diff = AutoSimplify(ir::Sub::Make(lhs, rhs), *var_intervals_);
  VLOG(6) << "diff of " << ir::Sub::Make(lhs, rhs) << " = " << diff;
  if (diff.is_constant() && diff.get_constant() > 0) {
    return true;
  }
  ir::Expr diff_lower_bound = LowerBound(diff);
  VLOG(6) << "lower bound of " << diff << " = " << diff_lower_bound;
  if (diff_lower_bound.is_constant() && diff_lower_bound.get_constant() > 0) {
    return true;
  }
  return false;
}

bool SymbolicExprAnalyzer::CanProveLT(const ir::Expr& lhs,
                                      const ir::Expr& rhs) const {
  return CanProveGT(rhs, lhs);
}

class BoundReplacer : public ir::IRMutator<> {
 public:
  explicit BoundReplacer(cas_intervals_t* var_intervals, bool is_lower_bound)
      : var_intervals_(var_intervals), sign_(is_lower_bound) {}

  void operator()(ir::Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* var, ir::Expr* op) override {
    ir::Expr lower_bound = SymbolicExprLimit::negative_inf;
    ir::Expr upper_bound = SymbolicExprLimit::positive_inf;
    if (var_intervals_->count(var->name) != 0) {
      const CasInterval& interval = var_intervals_->at(var->name);
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
  cas_intervals_t* var_intervals_;
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

bool operator==(const SingleIntervalIntSet& lhs,
                const SingleIntervalIntSet& rhs) {
  cas_intervals_t merged_var_intervals = MergeVarIntervals(lhs, rhs);
  SymbolicExprAnalyzer analyzer(&merged_var_intervals);
  return analyzer.CanProveEQ(lhs.min_, rhs.min_) &&
         analyzer.CanProveEQ(lhs.max_, rhs.max_);
}

SingleIntervalIntSet Union(const SingleIntervalIntSet& a,
                           const SingleIntervalIntSet& b) {
  if (a.IsEmpty() || b.IsAll()) {
    return b;
  }
  if (b.IsEmpty() || a.IsAll()) {
    return a;
  }

  // May be relaxed when (a.max < b.min - 1) or (b.max < a.min - 1)
  cas_intervals_t merged_var_intervals = MergeVarIntervals(a, b);
  SymbolicExprAnalyzer analyzer(&merged_var_intervals);
  ir::Expr min = SymbolicExprLimit::positive_inf;
  ir::Expr max = SymbolicExprLimit::negative_inf;

  if (analyzer.CanProveLE(a.Min(), b.Min())) {
    min = a.Min();
  } else if (analyzer.CanProveGE(a.Min(), b.Min())) {
    min = b.Min();
  } else {
    min = ir::Min::Make(a.Min(), b.Min());
  }

  if (analyzer.CanProveGE(a.Max(), b.Max())) {
    max = a.Max();
  } else if (analyzer.CanProveLE(a.Max(), b.Max())) {
    max = b.Max();
  } else {
    max = ir::Max::Make(a.Max(), b.Max());
  }

  return SingleIntervalIntSet(min, max, std::move(merged_var_intervals));
}

SingleIntervalIntSet Intersect(const SingleIntervalIntSet& a,
                               const SingleIntervalIntSet& b) {
  if (a.IsAll() || b.IsEmpty()) {
    return b;
  }
  if (b.IsAll() || a.IsEmpty()) {
    return a;
  }

  cas_intervals_t merged_var_intervals = MergeVarIntervals(a, b);
  SymbolicExprAnalyzer analyzer(&merged_var_intervals);
  ir::Expr min = SymbolicExprLimit::positive_inf;
  ir::Expr max = SymbolicExprLimit::negative_inf;

  if (analyzer.CanProveLT(a.Max(), b.Min() - ir::Expr(1)) ||
      analyzer.CanProveLT(b.Max(), a.Min() - ir::Expr(1))) {
    return SingleIntervalIntSet(min, max);
  }

  if (analyzer.CanProveGE(a.Min(), b.Min())) {
    min = a.Min();
  } else if (analyzer.CanProveLE(a.Min(), b.Min())) {
    min = b.Min();
  } else {
    min = ir::Max::Make(a.Min(), b.Min());
  }

  if (analyzer.CanProveLE(a.Max(), b.Max())) {
    max = a.Max();
  } else if (analyzer.CanProveGE(a.Max(), b.Max())) {
    max = b.Max();
  } else {
    max = ir::Min::Make(a.Max(), b.Max());
  }

  return SingleIntervalIntSet(min, max, merged_var_intervals);
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

bool SingleIntervalIntSet::IsEmpty() const {
  return min_ == SymbolicExprLimit::positive_inf ||
         max_ == SymbolicExprLimit::negative_inf ||
         analyzer_.CanProveGT(min_, max_);
}

bool SingleIntervalIntSet::IsAll() const {
  return min_ == SymbolicExprLimit::negative_inf &&
         max_ == SymbolicExprLimit::positive_inf;
}

bool SingleIntervalIntSet::IsPoint() const {
  return analyzer_.CanProveEQ(min_, max_);
}

bool SingleIntervalIntSet::IsSubSet(const SingleIntervalIntSet& other) const {
  cas_intervals_t merged_var_intervals = MergeVarIntervals(*this, other);
  SymbolicExprAnalyzer analyzer(&merged_var_intervals);
  return analyzer.CanProveGE(min_, other.Min()) &&
         analyzer.CanProveLE(max_, other.Max());
}

bool SingleIntervalIntSet::IsSuperSet(const SingleIntervalIntSet& other) const {
  cas_intervals_t merged_var_intervals = MergeVarIntervals(*this, other);
  SymbolicExprAnalyzer analyzer(&merged_var_intervals);
  return analyzer.CanProveLE(min_, other.Min()) &&
         analyzer.CanProveGE(max_, other.Max());
}

}  // namespace common
}  // namespace cinn
