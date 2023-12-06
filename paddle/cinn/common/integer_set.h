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
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace common {

// A naive implementation of Symbolic Expression Analyzer
class SymbolicExprAnalyzer {
 public:
  explicit SymbolicExprAnalyzer(cas_intervals_t* var_intervals)
      : var_intervals_(var_intervals) {}

  bool CanProve(const ir::Expr& condition) const;
  bool CanProveEQ(const ir::Expr& lhs, const ir::Expr& rhs) const;
  bool CanProveNE(const ir::Expr& lhs, const ir::Expr& rhs) const;
  bool CanProveGE(const ir::Expr& lhs, const ir::Expr& rhs) const;
  bool CanProveLE(const ir::Expr& lhs, const ir::Expr& rhs) const;
  bool CanProveGT(const ir::Expr& lhs, const ir::Expr& rhs) const;
  bool CanProveLT(const ir::Expr& lhs, const ir::Expr& rhs) const;

  ir::Expr LowerBound(const ir::Expr& expr) const;
  ir::Expr UpperBound(const ir::Expr& expr) const;

 private:
  cas_intervals_t* var_intervals_;
};

// A helper struct to represent the positive infinity and negative infinity
struct SymbolicExprLimit {
  static ir::Expr positive_inf;
  static ir::Expr negative_inf;
};

// The set consisting of all integers in the interval from min to max
class SingleIntervalIntSet {
 public:
  explicit SingleIntervalIntSet(const ir::Expr& min,
                                const ir::Expr& max,
                                cas_intervals_t var_intervals = {});

  bool IsEmpty() const;
  bool IsAll() const;
  bool IsPoint() const;
  bool IsSubSet(const SingleIntervalIntSet& other) const;
  bool IsSuperSet(const SingleIntervalIntSet& other) const;

  friend bool operator==(const SingleIntervalIntSet& lhs,
                         const SingleIntervalIntSet& rhs);
  friend SingleIntervalIntSet Union(const SingleIntervalIntSet& a,
                                    const SingleIntervalIntSet& b);
  friend SingleIntervalIntSet Intersect(const SingleIntervalIntSet& a,
                                        const SingleIntervalIntSet& b);
  friend cas_intervals_t MergeVarIntervals(const SingleIntervalIntSet& a,
                                           const SingleIntervalIntSet& b);

  inline ir::Expr Min() const { return min_; }
  inline ir::Expr Max() const { return max_; }

 private:
  ir::Expr min_ = SymbolicExprLimit::positive_inf;
  ir::Expr max_ = SymbolicExprLimit::negative_inf;
  cas_intervals_t var_intervals_ = {};
  SymbolicExprAnalyzer analyzer_ = SymbolicExprAnalyzer(&var_intervals_);
};

}  // namespace common
}  // namespace cinn
