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

/**
 * This file includes some arithmetic utilities, such as simplifying/solving a
 * math equation/CINN expression.
 */
#pragma once

#include "paddle/cinn/ir/ir.h"  // NOLINT, Should be in front of other header files

#include <ginac/ginac.h>  // NOLINT

#include <limits>  // NOLINT
#include <map>     // NOLINT
#include <set>     // NOLINT
#include <string>  // NOLINT
#include <tuple>   // NOLINT

#ifdef As
#undef As
#endif

namespace cinn {
namespace common {

namespace ginac = GiNaC;

//! Tell whether the expression \p expr contains only simple math calculations,
//! like i*32+j is true, while Load(buf, i)+1 is not due to the Load Node is not
//! math related.
bool IsPureMath(Expr expr);

//! Tell whether the expression \p expr contains the expression \symbol, e.g.
//! i*32+32 contains `i`, it also contains `i+1`.
bool MathContainsSymbol(Expr expr, Var symbol);

//! Solve the equation \p lhs == \p rhs on symbol \p symbol.
std::tuple<Expr, bool /*positive*/> Solve(Expr lhs, Expr rhs, Var symbol);

//! Determine whether this expression \p expr calculates to be a zero.
bool MathIsZero(Expr expr);

int gcd(int a, int b);

/**
 * Helper to convert cinn::Expr to GiNaC::expr for some symbolic math analysis.
 */
struct ExprToGinacConverter {
  //! Convert CINN expression \p expr to GiNaC ex.
  ginac::ex operator()(Expr expr);

  //! Convert GiNaC ex back to CINN expression, should call operator() first.
  Expr GinacToExpr(const GiNaC::ex& ex);

  bool HasSymbol(const std::string& name) const {
    return repr_to_ginac_.count(name);
  }
  const ginac::symbol& GetSymbol(const std::string& name) const {
    return repr_to_ginac_.at(name);
  }

 private:
  std::string Repr(const Expr& expr);
  ginac::symbol CreateGinacSymbol(const std::string& repr);
  ginac::symbol CreateGinacSymbol(const ir::Expr& var);

  ginac::ex BuildHelper(ir::Expr expr);

  void RecordExpr(const ir::Expr& expr);

 private:
  std::map<std::string, ir::Expr> repr_to_expr_;
  std::map<std::string, ginac::symbol> repr_to_ginac_;
};

}  // namespace common
}  // namespace cinn
