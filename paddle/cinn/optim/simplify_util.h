// Copyright (c) 2025 CINN Authors. All Rights Reserved.
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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {
/*!
 * \brief Apply func `fleaf` into each leaf node of `expr`.
 * which leaf node is the most outside node that has TNode type.
 * \param expr The expression to be applied.
 * \param fleaf The function to be applied.
 */
template <typename TNode, typename FLeaf>
inline void UnpackReduction(const ir::IndexExpr &expr, FLeaf fleaf) {
  if (const TNode *node = expr.As<TNode>()) {
    UnpackReduction<TNode, FLeaf>(node->a(), fleaf);
    UnpackReduction<TNode, FLeaf>(node->b(), fleaf);
  } else {
    fleaf(expr);
  }
}

/*!
 * \brief Flatten the expression into a vector of expressions splited by `Add`
 * or `Mul`.
 *
 * For example (Add):
 * 1. `S0 + S1` ==> {S0, S1}
 * 2. `S0 + S1 * S2` ==> {S0, S1 * S2}
 * 3. `S0 + S1 * (S2 + S3)` ==> {S0, S1 * (S2 + S3)}
 *
 * \param lhs The left hand side expression to be compared.
 * \param rhs The right hand side expression to be compared.
 * \return A boolean value indicating whether the priority of `lhs` is higher
 * than `rhs`.
 */
template <typename T>
inline std::vector<ir::IndexExpr> GetFlattenExprs(const ir::IndexExpr &expr) {
  std::vector<ir::IndexExpr> result;
  auto fcollect = [&](ir::IndexExpr val) { result.push_back(val); };
  UnpackReduction<T>(expr, fcollect);
  return result;
}

/*!
 * \brief Compare the priority of the two expressions. this func follows the
 * above rules:
 * 1. if lhs = var, rhs = const,    return true;
 * 2. if lhs = const, rhs = var,    return false;
 * 3. if lhs = var, rhs = var,      return lhs_var_name <= lhs_var_name;
 * 4. if lhs.length > rhs.length,   return true;
 * 5. if lhs.length == rhs.length,  return lhs_type <= rhs_type; (Add < Mul <
 * Div < Mod)
 * 6. if lhs.length < rhs.length    return false;
 *
 * For example:
 * 1. `ComparePriority(S0, 2)` return true;
 * 2. `ComparePriority(S0, S0)` return true;
 * 2. `ComparePriority(S0, S1)` return false;
 * 3. `ComparePriority(S0, S1 + 1)` return false;
 * 4. `ComparePriority(S0 % 2, S1 + 1)` return false;
 *
 * \param lhs The left hand side expression to be compared.
 * \param rhs The right hand side expression to be compared.
 * \return A boolean value indicating whether the priority of `lhs` is higher
 * than `rhs`.
 */
bool ComparePriority(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs);

/*!
 * \brief Determines whether there are sub-parts in the `expr` that can be
 * simplified by `Add` operation with the input `symbol`. If true is returned,
 * the operation will be attempted on each subpart in outer
 * `SimplifySymbolicAdd` function.
 *
 * For example:
 * 1. `IsSumPartialBySymbol(5, S0)` return false;
 * 2. `IsSumPartialBySymbol(S0, S0)` return true;
 * 3. `IsSumPartialBySymbol(S0 + S1, S1)` return true;
 * 4. `IsSumPartialBySymbol(S0 * 5 + S1, S0)` return true;
 * 5. `IsSumPartialBySymbol(S0 / 3, S0)` return true;
 * 6. `IsSumPartialBySymbol(S0 / 3 + S1, S0)` return true;
 * 7. `IsSumPartialBySymbol(S0 % 3, S0)` return false;
 *
 * Note: For performance reasons, special patterns will not be matched here.
 * This can be allowed for extreme optimization.
 * For example:
 * `IsSumPartialBySymbol((S0 + S1 / 5 * 25) / 5, S1 % 5)` return false;
 *
 * \param expr The expression to be checked.
 * \param symbol  The symbol to be checked.
 * \return True means there are sub-parts in the `expr` that can be simplified.
 */
bool IsSumPartialBySymbol(const ir::IndexExpr &expr,
                          const ir::IndexExpr &symbol);

/*!
 * \brief Simplify the `lhs` by symbol `sym`. Usually run after
 * `IsSumPartialBySymbol`
 *
 * \param lhs The expression to be simplified.
 * \param sym  The symbol to be checked.
 *    it may be `i, j ..` or  `S0, S1 ..` or other symbolic expr.
 * \param outer_mul_factor The scale of symbolic expr.
 *    e.g. `S0 * 4` ===> sym == S0, outer_mul_factor == 4
 * \return The expr after simplification.
 */
ir::IndexExpr SimplifySymbolicAdd(
    const ir::IndexExpr &lhs,
    const ir::IndexExpr &sym,
    const ir::IndexExpr &outer_mul_factor = ir::IndexExpr(1));

/*!
 * \brief Determines whether there are sub-parts in the `expr` that can be
 * simplified by `Div` operation with the input `symbol`. If true is returned,
 * the operation will be attempted on each subpart in outer
 * `SimplifySymbolicDivide` function.
 *
 * For example:
 * 1. `IsDivisibleBySymbol(5, S0, div)` return false;
 * 2. `IsDivisibleBySymbol(S0, S0, div)` return true;
 * 3. `IsDivisibleBySymbol(S0 + S1, S1, div)` return false;
 * 4. `IsDivisibleBySymbol(S0 * 5 + S1 * S2, S0, div)` return true;
 * 5. `IsDivisibleBySymbol(S0 / 3, S0, div)` return true;
 * 6. `IsDivisibleBySymbol(S0 * 4 / 3, S0, div)` return true;
 * 7. `IsDivisibleBySymbol(S0 % 3, S0, div)` return false;
 * 8. `IsDivisibleBySymbol(S0 / 3, S0, mod)` return false;
 *
 * \param expr The expression to be checked.
 * \param symbol  The symbol to be checked.
 * \param ty ty is `Mod` or `Div`.
 * \return True means there are sub-parts in the `expr` that can be simplified.
 * \note this func dont deal the corner case, please use `ProveDivisible` for
 * exact result. e.g. `IsDivisibleBySymbol(f % S0 - f, S0, div)` is false
 */
bool IsDivisibleBySymbol(const ir::IndexExpr &expr,
                         const ir::IndexExpr &symbol,
                         const ir::IrNodeTy &ty);

/*!
 * \brief Simplify the `lhs` by symbol `sym`. Usually run after
 * `IsDivisibleBySymbol`
 *
 * \param lhs The expression to be simplified.
 * \param sym  The symbol to be checked.
 *    it may be `i, j ..` or  `S0, S1 ..` or other symbolic expr.
 * \param ty ty is `Mod` or `Div`.
 * \return The expr after simplification.
 */
ir::IndexExpr SimplifySymbolicDivide(const ir::IndexExpr &lhs,
                                     const ir::IndexExpr &sym,
                                     const ir::IrNodeTy &ty);

/*!
 * \brief Determine whether `lhs` is divisible by `rhs`, regardless of whether
 * `rhs` is a constant or a symbol.
 * \param lhs lhs is dividend.
 * \param rhs rhs is divisor.
 * \return A boolean value indicating whether the `lhs` is divisible by `rhs`
 */
bool ProveDivisible(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs);

/*!
 * \brief Judge whether `candidate` is a negated index expression.
 * \param candidate The expression to be checked.
 * \param expr The positive part
 * \return A boolean value indicating whether `candidate` is negative.
 */
bool IsNegatedIndexExpr(const ir::IndexExpr &candidate,
                        ir::IndexExpr &expr);  // NOLINT

/*!
 * \brief Construct index expression by node type with or without simplify.
 * \param ty The node type of index expression.
 * \param lhs left operand.
 * \param rhs right operand.
 * \param simplify_flag Whether to simplify the result.
 * \return The constructed index expression.
 */
ir::IndexExpr ConstructIndexExprByNodeType(const ir::IrNodeTy &ty,
                                           const ir::IndexExpr &lhs,
                                           const ir::IndexExpr &rhs,
                                           bool simplify_flag = true);

/*!
 * \brief Change the sequence of `Div` and `Mod` in index expression.
 * Mathematical formula: `(a / b) % c = (a % (b * c)) / b`
 * For example:
 * 1. i / 4 % 8 => i % 32 / 4
 * 2. i / S0 % S1 => i % (S0 * S1) / S0
 * 3. (i * 32 + j) / 4 % 8 => (i * 32 + j) % 32 / 4
 *
 * \param expr The `IndexExpr` to be change
 * \return `IndexExpr` after change.
 */
ir::IndexExpr ChangeSeqOfDivMod(const ir::IndexExpr &expr);

/*!
 * \brief Judge type of `expr` is valid type of `IndexExpr` or not.
 * \param expr The expression to be checked.
 * \return A enum IndexType value indicating whether the type of `expr` is valid
 * IndexExpr.
 *
 * Note: Although load and cast are also legal IndexExpr, we need to know this
 * information in some scenarios.
 */
ir::IndexExpr::IndexType VerifyIndex(const ir::Expr &expr);

/*!
 * \brief The multiplication in rhs is broken down and each sub-part is
 * independently determined to be divisible.
 * \param lhs The dividend.
 * \param rhs The divisor.
 * \param ty  ty is `Mod` or `Div`.
 * \return A optional index expression indicating whether the `lhs`
 * is divisible, nullopt indicating not divisible.
 *
 * For example:
 * 1. i * S0 * S1 * S2 / (S0 * S1) ==> i / S2
 * 2. i * S0 * S1 / S0 ==> i * S1
 * 3. i * S0 / (S0 + 1) ==> nullopt
 */
std::optional<ir::IndexExpr> DivByPartMul(const ir::IndexExpr &lhs,
                                          const ir::IndexExpr &rhs,
                                          ir::IrNodeTy ty);

/*!
 * \brief Simplify complex modulo expressions.
 * \param lhs The dividend.
 * \param rhs The divisor.
 * \return A optional index expression indicating whether simplified
 *
 * For example:
 * 1. (i / S0 * S0 + i % (S0 * S1)) % S0 ==> i % S0
 * 2. (i / S0 * S0 * S1 + i % (S0 * S1 * S2)) % (S0 * S1) ==> i % (S0 * S1)
 * 3. i % (S0 * S1) % S0 ==> i % S0
 * 4. i * S0 * S1 % (S0 * S1) ==> 0
 */
std::optional<ir::IndexExpr> SimplifyComplexMod(const ir::IndexExpr &lhs,
                                                const ir::IndexExpr &rhs);

/*!
 * \brief Check whether the expression matches the pattern.
 * \param expr The expression to be checked.
 * \param pattern The pattern to be matched. which includes some variables.
 * \param map return the matched variables.
 * \return A boolean value indicating whether `expr` is matched.
 *
 * For example:
 * 1. (i / S0 * S0 + i % (S0 * S1)) % S0 matched by a / b * b + a % (b * c)
 *    with map = {a: i, b: S0, c: S1}
 * 2. S0 + 5 matched by a + 5 with map = {a: S0, b: 5}
 *
 * Note: a * b and b * a is two different pattern.
 */
bool CheckPattern(const ir::IndexExpr &expr,
                  const ir::IndexExpr &pattern,
                  std::unordered_map<std::string, ir::IndexExpr> *map);

// TODO(liujinnan): Delete historical `simplify func` related files, temporary
// placement of tool functions that are still in use, remove it in the future.
bool IsPureMath(Expr expr);

/*!
 * \brief Parse the expression from string to Expr.
 * \param expr_str The expression to be checked.
 * \return A Expr parsed from string.
 *
 * For example:
 * 1. ParseExpressionFromString("a + b * c") return Add(Var(a), Mul(Var(b),
 * Var(c)))
 * 2. ParseExpressionFromString("a + 10") return Add(Var(a), IntImm(10)))
 */
ir::Expr ParseExpressionFromString(const std::string &expr_str);

/*!
 * \brief Check whether the expression matches the pattern.
 * \param expr The expression to be checked.
 * \param pattern_str The pattern string to be matched.
 * \param condition A optional condition function to further check the matched
 * \return A optional map indicating the matched variables.
 *
 * For example:
 * 1. i / S0 * S0 + i % (S0 * S1) matched by a / b * b + a % (b * c)
 *    return map = {a: i, b: S0, c: S1}
 * 2. i / (S0 * S1) * S0 + i % (S0 * S1) / S1 matched by a / f * b + a % f / c
 * with optional condition f = a * b, return map = {a: i, b: S0, c: S1, f: S0 *
 * S1}
 * 3. S0 + 5 matched by a + 5 return map = {a: S0, b: 5}
 */
std::optional<std::unordered_map<std::string, ir::IndexExpr>> MatchPattern(
    const ir::IndexExpr &expr,
    const std::string &pattern_str,
    const std::function<bool(
        const std::unordered_map<std::string, ir::IndexExpr> &)> &condition =
        nullptr);

/*!
 * \brief Simplify IndexExpr with bound information.
 * For example:
 *        x % S0 ==> x if x < S0
 *        x / S0 ==> 0 if x < S0
 *
 * \param expr The `IndexExpr` to be simplified.
 * \return `IndexExpr` after simplification.
 */
ir::IndexExpr BoundSimplify(const ir::IndexExpr &expr);

/*!
 * \brief Simplify IndexExpr with broadcastable information.
 * For example:
 *        x % cinn_max(cinn_max(cinn_max(S0, S10), S20), S30))
 *          % cinn_max(S10, S30) ==> x % cinn_max(S10, S30),
 *        if broadcastable(S0, S10, S20, S30).
 * Note: The following conditions must be met:
 * 1. Two consecutive modular operations.
 * 2. The first modulus is broadcastable.
 * 3. The second modulus is a subset of the first modulus.
 *
 * \param expr The `IndexExpr` to be simplified.
 * \return `IndexExpr` after simplification.
 */
ir::IndexExpr BroadcastSimplify(const ir::IndexExpr &expr);
}  // namespace optim
}  // namespace cinn
