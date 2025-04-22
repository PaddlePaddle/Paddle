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

#include "paddle/cinn/optim/simplify_util.h"
#include <algorithm>
#include <stack>
#include <unordered_set>

#include "paddle/cinn/common/const_fold.h"
#include "paddle/cinn/common/shape_constraint.h"
#include "paddle/cinn/common/simplify_special_pattern.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace optim {

bool ComparePriority(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs) {
  if (lhs.node_type() == ir::IrNodeTy::IntImm &&
      rhs.node_type() != ir::IrNodeTy::IntImm)
    return false;
  if (rhs.node_type() == ir::IrNodeTy::IntImm &&
      lhs.node_type() != ir::IrNodeTy::IntImm)
    return true;
  if (auto lhsVar = lhs.As<ir::_Var_>())
    if (auto rhsVar = rhs.As<ir::_Var_>())
      return std::make_tuple(lhsVar->name.length(), lhsVar->name) <=
             std::make_tuple(rhsVar->name.length(), rhsVar->name);
  auto lhsLen = lhs.length();
  auto rhsLen = rhs.length();
  if (lhsLen < rhsLen) return false;
  // Add < Mul < Div < Mod < Min < Max < Cast < Load.
  else if (lhsLen == rhsLen)
    return lhs.node_type() <= rhs.node_type();
  else
    return true;
}

bool IsSumPartialBySymbol(const ir::IndexExpr &expr,
                          const ir::IndexExpr &symbol) {
  if (expr == symbol) return true;
  // TODO(liujinnan): Check Ty
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm: {
      return false;
    }
    case ir::IrNodeTy::_Var_:
      return expr == symbol;
    case ir::IrNodeTy::Add:
      return IsSumPartialBySymbol(expr.operand(0), symbol) ||
             IsSumPartialBySymbol(expr.operand(1), symbol);
    case ir::IrNodeTy::Mul: {
      if (expr.operand(1).is_constant() && expr.operand(1).get_constant() == -1)
        return IsSumPartialBySymbol(expr.operand(0), symbol);
      else
        return expr.operand(0) == symbol || expr.operand(1) == symbol;
    }

    case ir::IrNodeTy::Div: {
      return IsSumPartialBySymbol(expr.operand(0), symbol);
    }
    case ir::IrNodeTy::Mod:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Load:
    case ir::IrNodeTy::Cast:
      return false;
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in IsSumPartialBySymbol which is: %s",
          expr));
  }
}
ir::IndexExpr SimplifySymbolicAdd(const ir::IndexExpr &lhs,
                                  const ir::IndexExpr &sym,
                                  const ir::IndexExpr &outer_mul_factor) {
  if (lhs == sym) return sym * (outer_mul_factor + ir::IndexExpr(1));
  switch (lhs.node_type()) {
    case ir::IrNodeTy::IntImm: {
      auto imm = lhs.As<ir::IntImm>();
      if (imm->value != 0)
        PADDLE_THROW(::common::errors::Fatal("Error in SimplifySymbolicAdd!"));
      return ir::IndexExpr(0);
    }
    case ir::IrNodeTy::_Var_: {
      return sym * (outer_mul_factor + ir::IndexExpr(1));
    }
    case ir::IrNodeTy::Add: {
      if (!IsSumPartialBySymbol(lhs.operand(0), sym))
        return lhs.operand(0) +
               SimplifySymbolicAdd(lhs.operand(1), sym, outer_mul_factor);
      return SimplifySymbolicAdd(lhs.operand(0), sym, outer_mul_factor) +
             lhs.operand(1);
    }
    case ir::IrNodeTy::Mul: {
      if (lhs.operand(1).is_constant() && lhs.operand(1).as_int64() == -1) {
        return SimplifySymbolicAdd(lhs.operand(0), sym, -outer_mul_factor) *
               lhs.operand(1);
      }
      if (lhs.operand(0) == sym)
        return lhs.operand(0) * (lhs.operand(1) + outer_mul_factor);
      return (lhs.operand(0) + outer_mul_factor) * lhs.operand(1);
    }
    case ir::IrNodeTy::Mod:
      PADDLE_THROW(::common::errors::Fatal("Error in SimplifySymbolicAdd!"));
    case ir::IrNodeTy::Div: {
      return SimplifySymbolicAdd(
                 lhs.operand(0), sym, lhs.operand(1) * outer_mul_factor) /
             lhs.operand(1);
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of lhs in SimplifySymbolicAdd which is: %s", lhs));
  }
}

bool IsDivisibleBySymbol(const ir::IndexExpr &expr,
                         const ir::IndexExpr &symbol,
                         const ir::IrNodeTy &ty) {
  if (expr == symbol) return true;
  // TODO(liujinnan): Check Ty
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm: {
      auto imm = expr.As<ir::IntImm>();
      return imm->value == 0;
    }
    case ir::IrNodeTy::_Var_:
      return expr == symbol;
    case ir::IrNodeTy::Add:
      return IsDivisibleBySymbol(expr.operand(0), symbol, ty) &&
             IsDivisibleBySymbol(expr.operand(1), symbol, ty);
    case ir::IrNodeTy::Mul:
      // Because (S0 / 7 * 100) / S0 is not divisible by S0, so we push
      // `expr.node_type()` into third parameter.
      return IsDivisibleBySymbol(expr.operand(0), symbol, expr.node_type()) ||
             IsDivisibleBySymbol(expr.operand(1), symbol, expr.node_type());
    case ir::IrNodeTy::Mod:
      // Because S0 % 3 + S0 % 5 is not divisible by S0, so we push
      // `expr.node_type()` into third parameter.
      return IsDivisibleBySymbol(expr.operand(0), symbol, expr.node_type()) &&
             IsDivisibleBySymbol(expr.operand(1), symbol, expr.node_type());
    case ir::IrNodeTy::Div: {
      if (ty != expr.node_type()) return false;
      return IsDivisibleBySymbol(expr.operand(0), symbol, expr.node_type());
    }
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Load:
    case ir::IrNodeTy::Cast:
      return false;
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in IsDivisibleBySymbol which is: %s",
          expr));
  }
}

ir::IndexExpr SimplifySymbolicDivide(const ir::IndexExpr &lhs,
                                     const ir::IndexExpr &sym,
                                     const ir::IrNodeTy &ty) {
  if (lhs == sym) return ir::IndexExpr(1);
  switch (lhs.node_type()) {
    case ir::IrNodeTy::IntImm: {
      auto imm = lhs.As<ir::IntImm>();
      if (imm->value != 0)
        PADDLE_THROW(
            ::common::errors::Fatal("Error in SimplifySymbolicDivide!"));
      return ir::IndexExpr(0);
    }
    case ir::IrNodeTy::_Var_:
      return ir::IndexExpr(1);
    case ir::IrNodeTy::Add:
      return SimplifySymbolicDivide(lhs.operand(0), sym, ty) +
             SimplifySymbolicDivide(lhs.operand(1), sym, ty);
    case ir::IrNodeTy::Mul: {
      if (!IsDivisibleBySymbol(lhs.operand(0), sym, ty))
        return lhs.operand(0) * SimplifySymbolicDivide(lhs.operand(1), sym, ty);
      return SimplifySymbolicDivide(lhs.operand(0), sym, ty) * lhs.operand(1);
    }
    case ir::IrNodeTy::Mod:
      return SimplifySymbolicDivide(lhs.operand(0), sym, lhs.node_type()) %
             SimplifySymbolicDivide(lhs.operand(1), sym, lhs.node_type());
    case ir::IrNodeTy::Div: {
      return SimplifySymbolicDivide(lhs.operand(0), sym, lhs.node_type()) /
             lhs.operand(1);
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of lhs in SimplifySymbolicDivide which is: %s",
          lhs));
  }
}

bool ProveDivisible(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs) {
  if (IsZero(lhs % rhs)) return true;
  if (IsZero(optim::ArithSimplify(lhs % rhs))) return true;
  return false;
}

bool IsNegatedIndexExpr(const ir::IndexExpr &candidate,
                        ir::IndexExpr &expr) {  // NOLINT
  if (auto mul = candidate.As<ir::Mul>()) {
    if (mul->b().is_constant() && mul->b().get_constant() == -1) {
      expr = mul->a();
      return true;
    }
  }
  return false;
}

ir::IndexExpr::IndexType VerifyIndex(const ir::Expr &expr) {
  switch (expr.node_type()) {
    case ir::IrNodeTy::_Var_:
    case ir::IrNodeTy::IntImm: {
      return expr.type().is_index_type() ? ir::IndexExpr::IndexType::kValid
                                         : ir::IndexExpr::IndexType::kInvalid;
    }
    case ir::IrNodeTy::Load: {
      return expr.type().is_index_type() ? ir::IndexExpr::IndexType::kLoad
                                         : ir::IndexExpr::IndexType::kInvalid;
    }
    case ir::IrNodeTy::Cast: {
      ir::IndexExpr::IndexType result = VerifyIndex(expr->operand(0));
      return result != ir::IndexExpr::IndexType::kInvalid &&
                     expr.type().is_index_type()
                 ? ir::IndexExpr::IndexType::kCast
                 : ir::IndexExpr::IndexType::kInvalid;
    }
    case ir::IrNodeTy::Add:
    case ir::IrNodeTy::Sub:
    case ir::IrNodeTy::Mul:
    case ir::IrNodeTy::Div:
    case ir::IrNodeTy::Mod:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Min: {
      ir::IndexExpr::IndexType left = VerifyIndex(expr->operand(0));
      ir::IndexExpr::IndexType right = VerifyIndex(expr->operand(1));
      if (left == ir::IndexExpr::IndexType::kInvalid ||
          right == ir::IndexExpr::IndexType::kInvalid)
        return ir::IndexExpr::IndexType::kInvalid;
      return std::max(left, right);
    }
  }
  return ir::IndexExpr::IndexType::kInvalid;
}

ir::IndexExpr ConstructIndexExprByNodeType(const ir::IrNodeTy &ty,
                                           const ir::IndexExpr &lhs,
                                           const ir::IndexExpr &rhs,
                                           bool simplify_flag) {
  switch (ty) {
    case ir::IrNodeTy::Add:
      return simplify_flag ? lhs + rhs : ir::Add::Make(lhs, rhs);
    case ir::IrNodeTy::Sub:
      return simplify_flag ? lhs - rhs : ir::Sub::Make(lhs, rhs);
    case ir::IrNodeTy::Mul:
      return simplify_flag ? lhs * rhs : ir::Mul::Make(lhs, rhs);
    case ir::IrNodeTy::Div:
      return simplify_flag ? lhs / rhs : ir::Div::Make(lhs, rhs);
    case ir::IrNodeTy::Mod:
      return simplify_flag ? lhs % rhs : ir::Mod::Make(lhs, rhs);
    case ir::IrNodeTy::Min:
      return ir::Min::Make(lhs, rhs);
    case ir::IrNodeTy::Max:
      return ir::Max::Make(lhs, rhs);
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type in Constructir::IndexExprByNodeType, which is: %s",
          ty));
  }
}

ir::IndexExpr ChangeSeqOfDivMod(const ir::IndexExpr &expr) {
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm:
    case ir::IrNodeTy::_Var_:
    case ir::IrNodeTy::Cast:
    case ir::IrNodeTy::Load: {
      return expr;
    }
    case ir::IrNodeTy::Add:
    case ir::IrNodeTy::Sub:
    case ir::IrNodeTy::Mul:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Div: {
      auto lhs = ChangeSeqOfDivMod(expr.operand(0));
      auto rhs = ChangeSeqOfDivMod(expr.operand(1));
      return ConstructIndexExprByNodeType(expr.node_type(), lhs, rhs, false);
    }
    case ir::IrNodeTy::Mod: {
      if (expr.operand(0).node_type() == ir::IrNodeTy::Div) {
        auto div_lhs = ChangeSeqOfDivMod(expr.operand(0).operand(0));
        auto div_rhs = ChangeSeqOfDivMod(expr.operand(0).operand(1));
        auto mod_rhs = ChangeSeqOfDivMod(expr.operand(1));
        return div_lhs % (div_rhs * mod_rhs) / div_rhs;
      } else {
        auto lhs = ChangeSeqOfDivMod(expr.operand(0));
        auto rhs = ChangeSeqOfDivMod(expr.operand(1));
        if (lhs.node_type() == ir::IrNodeTy::Div) {
          return (lhs.operand(0) % (lhs.operand(1) * rhs)) / lhs.operand(1);
        }
        return ConstructIndexExprByNodeType(expr.node_type(), lhs, rhs, false);
      }
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in ChangeSeqOfDivMod which is: %s", expr));
  }
}
std::optional<ir::IndexExpr> DivByPartMul(const ir::IndexExpr &lhs,
                                          const ir::IndexExpr &rhs,
                                          ir::IrNodeTy ty) {
  std::vector<ir::IndexExpr> elems = GetFlattenExprs<ir::Mul>(rhs);

  ir::IndexExpr result = ir::ir_utils::IRCopy(lhs);

  for (const auto &elem : elems) {
    if (IsDivisibleBySymbol(result, elem, ty)) {
      result = SimplifySymbolicDivide(result, elem, ty);
    } else {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<ir::IndexExpr> SimplifyComplexMod(const ir::IndexExpr &lhs,
                                                const ir::IndexExpr &rhs) {
  if (lhs == rhs) return ir::IndexExpr(lhs.type(), 0);
  switch (lhs.node_type()) {
    case ir::IrNodeTy::Add: {
      auto simplify_lhs = SimplifyComplexMod(lhs.operand(0), rhs);
      auto simplify_rhs = SimplifyComplexMod(lhs.operand(1), rhs);
      if (simplify_lhs.has_value() && simplify_rhs.has_value())
        return (simplify_lhs.value() + simplify_rhs.value());
      return std::nullopt;
    }
    case ir::IrNodeTy::Mul: {
      // (S0 % 4 * S1 % 8) % 4 != S0 % 4 * S1 % 4;
      if (DivByPartMul(lhs, rhs, ir::IrNodeTy::Mod))
        return ir::IndexExpr(lhs.type(), 0);
      return std::nullopt;
    }
    case ir::IrNodeTy::Div:
    case ir::IrNodeTy::IntImm:
    case ir::IrNodeTy::_Var_:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Load:
    case ir::IrNodeTy::Cast: {
      return std::nullopt;
    }
    case ir::IrNodeTy::Mod: {
      if (DivByPartMul(lhs.operand(1), rhs, ir::IrNodeTy::Mod)) {
        return lhs.operand(0) % rhs;
      }
      return std::nullopt;
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in SimplifyComplexMod which is: %s", lhs));
  }
  return std::nullopt;
}

bool CheckPattern(const ir::IndexExpr &expr,
                  const ir::IndexExpr &pattern,
                  std::unordered_map<std::string, ir::IndexExpr> *map) {
  // pattern may include Var to match any expr.
  if (expr.node_type() != pattern.node_type() &&
      pattern.node_type() != ir::IrNodeTy::_Var_)
    return false;
  switch (pattern.node_type()) {
    case ir::IrNodeTy::Add:
    case ir::IrNodeTy::Sub:
    case ir::IrNodeTy::Mul:
    case ir::IrNodeTy::Div:
    case ir::IrNodeTy::Mod:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max: {
      return CheckPattern(expr.operand(0), pattern.operand(0), map) &&
             CheckPattern(expr.operand(1), pattern.operand(1), map);
    }
    case ir::IrNodeTy::_Var_: {
      auto it = map->find(pattern.As<ir::_Var_>()->name);
      if (it != map->end()) {
        return expr == it->second;
      } else {
        map->insert(std::make_pair(pattern.As<ir::_Var_>()->name, expr));
        return true;
      }
    }
    case ir::IrNodeTy::IntImm: {
      return expr.As<ir::IntImm>()->value == pattern.As<ir::IntImm>()->value;
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in CheckPattern which is: %s", expr));
  }

  return false;
}

bool IsPureMath(Expr expr) {
  std::set<ir::IrNodeTy> valid_node_tys({
      ir::IrNodeTy ::_Var_,
      ir::IrNodeTy ::IntImm,
      ir::IrNodeTy ::Sum,
      ir::IrNodeTy ::Product,
      ir::IrNodeTy ::FracOp,
      ir::IrNodeTy ::FloatImm,
      ir::IrNodeTy ::Add,
      ir::IrNodeTy ::Sub,
      ir::IrNodeTy ::Div,
      ir::IrNodeTy ::Mul,
      ir::IrNodeTy::Mod,
      ir::IrNodeTy ::Minus,
  });

  auto complex_nodes = ir::ir_utils::CollectIRNodes(expr, [&](const Expr *n) {
    return !valid_node_tys.count(n->node_type());
  });
#ifdef CINN_DEBUG
  for (auto &node : complex_nodes) {
    VLOG(3) << "Found " << node->node_type() << " " << Expr(node);
  }
#endif
  return complex_nodes.empty();
}

/*!
 * \brief Index Token in Tokenizer and Parser
 */
struct IndexToken {
  enum class TokenType {
    kNumber,
    kVar,
    kPlus,
    kMinus,
    kMultiply,
    kDivide,
    kModulo,
    kLeftParen,
    kRightParen,
    kEnd
  };

  TokenType type;
  std::string value;

  explicit IndexToken(TokenType t, const std::string &v = "")
      : type(t), value(v) {}
};

/*!
 * \brief Tokenizer for IndexExpr, split the input string into IndexToken.
 */
class Tokenizer {
 public:
  explicit Tokenizer(const std::string &in) : input(in), pos(0) {}
  // generate IndexToken for the next `pos`. it supports the following:
  // 1. Number: 123, 1234...
  // 2. Variable: a, b, a_1, aa, f1...
  // 3. Operator: +, -, *, /, %, (, )
  // 4. Whitespace
  IndexToken NextToken() {
    // skip whitespace
    while (pos < input.size() && std::isspace(input[pos])) {
      pos++;
    }
    // check if we reached the end of the input
    if (pos >= input.size()) {
      return IndexToken(IndexToken::TokenType::kEnd);
    }

    char c = input[pos++];

    // deal with number (0, 1, 11, 123...) not support float.
    if (std::isdigit(c)) {
      std::string num;
      num += c;
      while (pos < input.size() && std::isdigit(input[pos])) {
        num += input[pos++];
      }
      return IndexToken(IndexToken::TokenType::kNumber, num);
    }

    // deal with variable name (a, b, a1, a123, a_1...).
    if (std::isalpha(c) || input[pos] == '_') {
      std::string var;
      var += c;
      while (pos < input.size() &&
             (std::isalnum(input[pos]) || input[pos] == '_')) {
        var += input[pos++];
      }
      return IndexToken(IndexToken::TokenType::kVar, var);
    }

    // deal with operator {+, -, *, /, %, '(', ')'}.
    switch (c) {
      case '+':
        return IndexToken(IndexToken::TokenType::kPlus);
      case '-':
        return IndexToken(IndexToken::TokenType::kMinus);
      case '*':
        return IndexToken(IndexToken::TokenType::kMultiply);
      case '/':
        return IndexToken(IndexToken::TokenType::kDivide);
      case '%':
        return IndexToken(IndexToken::TokenType::kModulo);
      case '(':
        return IndexToken(IndexToken::TokenType::kLeftParen);
      case ')':
        return IndexToken(IndexToken::TokenType::kRightParen);
      default:
        PADDLE_THROW(::common::errors::InvalidArgument(
            "Tokenizer Unexpected character: %s", c));
    }
  }

 private:
  const std::string &input;
  size_t pos;
};

/*!
 * \brief Parser for IndexExpr, parse the input string into ir::Expr.
 */
class Parser {
 public:
  explicit Parser(const std::string &input)
      : tokenizer(input), currentToken(tokenizer.NextToken()) {}
  ir::Expr Parse() { return ParseExpression(); }

 private:
  void Advance() { currentToken = tokenizer.NextToken(); }

  // Processing addition and subtraction expressions, with the lowest priority.
  ir::Expr ParseExpression() {
    auto left = ParseTerm();

    while (currentToken.type == IndexToken::TokenType::kPlus ||
           currentToken.type == IndexToken::TokenType::kMinus) {
      auto op = currentToken.type;
      Advance();
      auto right = ParseTerm();

      if (op == IndexToken::TokenType::kPlus) {
        left = ir::Add::Make(left, right);
      } else {
        left = ir::Sub::Make(left, right);
      }
    }

    return left;
  }

  // Process multiplication, division and modulo expressions, with higher
  // priority than addition and subtraction, and the parsing result appears as
  // one Term. e.g. a * b + c, a * b is a Term.
  ir::Expr ParseTerm() {
    auto left = ParseFactor();
    while (currentToken.type == IndexToken::TokenType::kMultiply ||
           currentToken.type == IndexToken::TokenType::kDivide ||
           currentToken.type == IndexToken::TokenType::kModulo) {
      auto op = currentToken.type;
      Advance();
      auto right = ParseFactor();

      if (op == IndexToken::TokenType::kMultiply) {
        left = ir::Mul::Make(left, right);
      } else if (op == IndexToken::TokenType::kDivide) {
        left = ir::Div::Make(left, right);
      } else {
        left = ir::Mod::Make(left, right);
      }
    }

    return left;
  }

  // Process numeric, variables and brackets, with the highest priority, as
  // parameters for each item.
  ir::Expr ParseFactor() {
    if (currentToken.type == IndexToken::TokenType::kNumber) {
      int value = std::stoi(currentToken.value);
      Advance();
      return ir::Expr(value);
    } else if (currentToken.type == IndexToken::TokenType::kVar) {
      auto var_name = currentToken.value;
      Advance();
      return GetOrCreateVar(var_name);
    } else if (currentToken.type == IndexToken::TokenType::kLeftParen) {
      Advance();
      auto expr = ParseExpression();

      if (currentToken.type != IndexToken::TokenType::kRightParen) {
        PADDLE_THROW(::common::errors::InvalidArgument(
            "Parser Expected ')', because of '(' in before."));
      }

      Advance();
      return expr;
    } else {
      PADDLE_THROW(
          ::common::errors::InvalidArgument("Parser Unexpected IndexToken"));
    }
  }
  ir::Expr GetOrCreateVar(const std::string &var_name) {
    if (vars.find(var_name) == vars.end()) {
      vars[var_name] = ir::Var(var_name);
    }
    return vars[var_name];
  }
  Tokenizer tokenizer;
  IndexToken currentToken;
  std::unordered_map<std::string, ir::Var> vars;
};

ir::Expr ParseExpressionFromString(const std::string &expr_str) {
  thread_local static std::unordered_map<std::string, ir::Expr> cache;
  auto it = cache.find(expr_str);
  if (it != cache.end()) {
    return it->second;
  }
  Parser parser(expr_str);
  auto result = parser.Parse();
  cache[expr_str] = result;

  return result;
}

std::optional<std::unordered_map<std::string, ir::IndexExpr>> MatchPattern(
    const ir::IndexExpr &expr,
    const std::string &pattern_str,
    const std::function<bool(
        const std::unordered_map<std::string, ir::IndexExpr> &)> &condition) {
  // Parse the pattern string into an IndexExpr
  ir::IndexExpr pattern = ParseExpressionFromString(pattern_str);

  std::unordered_map<std::string, ir::IndexExpr> map;

  if (CheckPattern(expr, pattern, &map)) {
    // Apply the condition if provided
    if (condition && !condition(map)) return std::nullopt;
    return map;
  }

  return std::nullopt;
}

ir::IndexExpr BoundSimplify(const ir::IndexExpr &expr) {
  // return expr if expr is not a division or modulo
  if (expr.node_type() != ir::IrNodeTy::Div &&
      expr.node_type() != ir::IrNodeTy::Mod)
    return expr;

  common::cas_intervals_t var_intervals =
      common::CollectVarIntervalsOfExprs({expr});
  common::SymbolicExprAnalyzer ana(var_intervals);
  // Because the SymbolicExprAnalyzer bound result is [lower, upper), `ProveLE`
  // is used here instead of `ProveLT`.
  auto canBeSimplified =
      ana.ProveLE(ana.UpperBound(expr.operand(0)), expr.operand(1));

  if (canBeSimplified.value_or(false)) {
    if (expr.node_type() == ir::IrNodeTy::Div) {
      return ir::IndexExpr(0);
    } else if (expr.node_type() == ir::IrNodeTy::Mod) {
      return expr.operand(0);
    }
  }
  return expr;
}

ir::IndexExpr BroadcastSimplify(const ir::IndexExpr &expr) {
  // Two consecutive modular operations.
  auto opt_map =
      MatchPattern(expr,
                   "f % a % b",
                   [](const std::unordered_map<std::string, ir::IndexExpr> &m) {
                     return m.at("a").node_type() == ir::IrNodeTy::Max ||
                            m.at("a").node_type() == ir::IrNodeTy::Mul;
                   });
  if (!opt_map) return expr;

  auto &map = opt_map.value();
  auto ll = map.at("f");
  auto lr = map.at("a");
  auto r = map.at("b");

  auto CanSimplifyMaxMod = [](const ir::IndexExpr &lr, const ir::IndexExpr &r) {
    auto lr_elems = GetFlattenExprs<ir::Max>(lr);
    auto r_elems = GetFlattenExprs<ir::Max>(r);

    // The second modulus is a subset of the first modulus.
    for (auto &&r_elem : r_elems) {
      if (std::find(lr_elems.begin(), lr_elems.end(), r_elem) == lr_elems.end())
        return false;
    }

    // The first modulus is broadcastable.
    auto &constraint = cinn::common::ShapeConstraintManager::Instance();
    return constraint.IsBroadcastable(lr_elems) ? true : false;
  };

  if (lr.node_type() == ir::IrNodeTy::Max) {
    if (CanSimplifyMaxMod(lr, r)) return ll % r;
    return expr;
  } else {
    std::unordered_map<ir::IndexExpr, int> r_elems;
    std::unordered_map<ir::IndexExpr, int> lr_elems;
    UnpackReduction<ir::Mul>(r, [&](ir::IndexExpr val) { r_elems[val]++; });
    UnpackReduction<ir::Mul>(lr, [&](ir::IndexExpr val) { lr_elems[val]++; });
    bool can_simplify = false;
    for (const auto &[r_first, r_second] : r_elems) {
      for (auto &[lr_first, lr_second] : lr_elems) {
        // Check equal relationship between the two operands.
        if (lr_first == r_first && lr_second >= r_second) {
          lr_second -= r_second;
          can_simplify = true;
          break;
        }
        // Check broadcastable relationship between the two operands.
        if (lr_first.node_type() == ir::IrNodeTy::Max &&
            CanSimplifyMaxMod(lr_first, r_first) && lr_second >= r_second) {
          lr_second -= r_second;
          can_simplify = true;
          break;
        }
      }
      if (!can_simplify) return expr;
    }
    return ll % r;
  }
}
}  // namespace optim
}  // namespace cinn
