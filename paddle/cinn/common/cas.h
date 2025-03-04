// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <absl/container/flat_hash_map.h>

#include <functional>
#include <string>
#include <vector>

#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

Expr AutoSimplify(
    const Expr& u,
    const absl::flat_hash_map<std::string, CasInterval>& var_intervals = {});

//! Simplify a CAS expression.
Expr CasSimplify(
    Expr u,
    const absl::flat_hash_map<std::string, CasInterval>& var_intervals = {});

namespace detail {

//! Whether to treat this expression as a symbol. e.g. Load, Min, Max are
//! treated as symbol to avoid confusing the CAS.
bool CASasSymbol(Expr expr);
//! Convert some nodes to CAS representation, e.g. convert Mul, Add to Product
//! and Sum.
Expr ConvertCinnToCAS(Expr expr);
//! Convert the CAS representation to CINN expression, e.g. convert Product and
//! Sum to Mul and Add.
Expr ConvertCasToCinn(Expr expr);
//! Tell whether this expression is acceptable by CAS.
bool IsExprCasCompatible(Expr expr);

struct ExprPosCmp {
  bool operator()(const Expr& a, const Expr& b);
};

struct CasSimplifyMutator {
  explicit CasSimplifyMutator(
      const absl::flat_hash_map<std::string, CasInterval> var_intervals)
      : var_intervals(var_intervals) {}

  Expr operator()(Expr u);

  Expr SimplifyRationalNumber(Expr u);
  Expr SimplifyPower(Expr u);
  Expr SimplifySum(Expr u);
  Expr SimplifyProduct(Expr a);
  Expr SimplifyMinAndMax(Expr a);
  Expr SimplifyCmp(Expr a);
  std::vector<Expr> SimplifyProductRec(const std::vector<Expr>& operands);
  std::vector<Expr> SimplifySumRec(const std::vector<Expr>& operands);
  Expr SimplifyMod(Expr u);
  Expr SimplifyFracOp(Expr expr);
  Expr SimplifyCond(Expr u);
  Expr FurtherSimplifyFracWithInterval(
      Expr expr,
      const absl::flat_hash_map<std::string, CasInterval>& var_intervals);
  Expr SimplifyIntegerPower(Expr u);
  void AddBaseAndSimplify(Expr* base, Expr bound);
  void UnfoldBound(Expr* lower_bound,
                   Expr* upper_bound,
                   Expr var,
                   bool unfold_const_bound = true);
  bool GetVarBound(Expr* lower_bound,
                   Expr* upper_bound,
                   Expr var,
                   bool unfold_const_bound = true);
  bool GetOperandBound(Expr* lower_bound,
                       Expr* upper_bound,
                       Expr var,
                       bool unfold_const_bound = true);
  bool GetSumBound(Expr* lower_bound,
                   Expr* upper_bound,
                   Expr sum,
                   bool unfold_const_bound = true);
  bool GetMinBound(Expr* lower_bound,
                   Expr* upper_bound,
                   Expr min,
                   bool unfold_const_bound = true);
  bool GetMaxBound(Expr* lower_bound,
                   Expr* upper_bound,
                   Expr max,
                   bool unfold_const_bound = true);
  bool GetExprBound(Expr* lower_bound,
                    Expr* upper_bound,
                    Expr min,
                    bool unfold_const_bound = true);
  bool SimplifySpecificSumMod(Expr* u, Expr a, Expr b);
  Expr SimplifySpecificSum(Expr u);

 private:
  std::vector<Expr> SimplifyBinaryProduct(Expr left, Expr right);
  std::vector<Expr> MergeProduct(const std::vector<Expr>& p,
                                 const std::vector<Expr>& q);

  std::vector<Expr> SimplifyBinarySum(Expr left, Expr right);
  std::vector<Expr> MergeSum(const std::vector<Expr>& p,
                             const std::vector<Expr>& q);
  std::vector<Expr> MergeExprs(
      const std::vector<Expr>& p,
      const std::vector<Expr>& q,
      const std::function<std::vector<Expr>(Expr, Expr)>& binary_merge);

  const absl::flat_hash_map<std::string, CasInterval> var_intervals;

  // Computation based on integer if set true(1/2 get 0), false if treat as
  // rational number in mathematics(1/2 is still 1/2), currently it only works
  // with true.
  bool int_compute_{true};
};

}  // namespace detail

}  // namespace common
}  // namespace cinn
