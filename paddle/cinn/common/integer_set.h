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
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace common {

cas_intervals_t CollectVarIntervalsOfExprs(const std::vector<ir::Expr>& exprs,
                                           bool is_lower_bound_zero = true);

// A naive implementation of Symbolic Expression Analyzer
class SymbolicExprAnalyzer {
 public:
  explicit SymbolicExprAnalyzer(const cas_intervals_t& var_intervals)
      : var_intervals_(var_intervals) {}
  SymbolicExprAnalyzer(const SymbolicExprAnalyzer&) = default;
  SymbolicExprAnalyzer(SymbolicExprAnalyzer&&) = default;
  SymbolicExprAnalyzer& operator=(const SymbolicExprAnalyzer&) = delete;
  SymbolicExprAnalyzer& operator=(SymbolicExprAnalyzer&&) = delete;

  // Try to prove the relationship of 2 symbolic expressions,
  // with the return value being optional.
  // If proven, return true. If falsified, return false.
  // If unable to prove or falsify, return nullopt.
  std::optional<bool> Prove(const ir::Expr& condition) const;
  std::optional<bool> ProveEQ(const ir::Expr& lhs, const ir::Expr& rhs) const;
  std::optional<bool> ProveNE(const ir::Expr& lhs, const ir::Expr& rhs) const;
  std::optional<bool> ProveGE(const ir::Expr& lhs, const ir::Expr& rhs) const;
  std::optional<bool> ProveLE(const ir::Expr& lhs, const ir::Expr& rhs) const;
  std::optional<bool> ProveGT(const ir::Expr& lhs, const ir::Expr& rhs) const;
  std::optional<bool> ProveLT(const ir::Expr& lhs, const ir::Expr& rhs) const;
  std::optional<bool> ProveDivisible(const ir::Expr& lhs,
                                     const ir::Expr& rhs) const;

  ir::Expr LowerBound(const ir::Expr& expr) const;
  ir::Expr UpperBound(const ir::Expr& expr) const;

 private:
  const cas_intervals_t& var_intervals_;
};

// A helper struct to represent the positive infinity and negative infinity
struct SymbolicExprLimit {
  static ir::Expr positive_inf;
  static ir::Expr negative_inf;
};

// The set consisting of all integers in the interval from min to max
class SingleIntervalIntSet {
 public:
  explicit SingleIntervalIntSet(
      const ir::Expr& min = SymbolicExprLimit::positive_inf,
      const ir::Expr& max = SymbolicExprLimit::negative_inf,
      cas_intervals_t var_intervals = {});
  SingleIntervalIntSet(const SingleIntervalIntSet& set) = default;
  SingleIntervalIntSet(SingleIntervalIntSet&& set) = default;
  SingleIntervalIntSet& operator=(const SingleIntervalIntSet& set) = default;
  SingleIntervalIntSet& operator=(SingleIntervalIntSet&& set) = default;

  // Try to prove or construct the relationship between two symbolic integer
  // sets, if unable to determine or construct, return nullopt.
  std::optional<bool> ProveEmpty() const;
  std::optional<bool> ProveAll() const;
  std::optional<bool> ProvePoint() const;
  std::optional<bool> ProveSubSet(const SingleIntervalIntSet& other) const;
  std::optional<bool> ProveSuperSet(const SingleIntervalIntSet& other) const;

  friend std::optional<bool> ProveEQ(const SingleIntervalIntSet& lhs,
                                     const SingleIntervalIntSet& rhs);
  friend std::optional<SingleIntervalIntSet> ProvedUnion(
      const SingleIntervalIntSet& a, const SingleIntervalIntSet& b);
  friend std::optional<SingleIntervalIntSet> ProvedIntersect(
      const SingleIntervalIntSet& a, const SingleIntervalIntSet& b);
  friend cas_intervals_t MergeVarIntervals(const SingleIntervalIntSet& a,
                                           const SingleIntervalIntSet& b);

  inline ir::Expr Min() const { return min_; }
  inline ir::Expr Max() const { return max_; }

 private:
  ir::Expr min_ = SymbolicExprLimit::positive_inf;
  ir::Expr max_ = SymbolicExprLimit::negative_inf;
  cas_intervals_t var_intervals_;
};

std::optional<bool> ProveEQ(const SingleIntervalIntSet& lhs,
                            const SingleIntervalIntSet& rhs);
std::optional<SingleIntervalIntSet> ProvedUnion(const SingleIntervalIntSet& a,
                                                const SingleIntervalIntSet& b);
std::optional<SingleIntervalIntSet> ProvedIntersect(
    const SingleIntervalIntSet& a, const SingleIntervalIntSet& b);
cas_intervals_t MergeVarIntervals(const SingleIntervalIntSet& a,
                                  const SingleIntervalIntSet& b);

ir::Expr EnhancedSimplifyModExpr(
    ir::Expr e,
    const absl::flat_hash_map<std::string, CasInterval>& var_intervals);

}  // namespace common
}  // namespace cinn
