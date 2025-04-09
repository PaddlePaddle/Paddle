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
#include <vector>

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace ir {

IndexExpr operator-(const IndexExpr& a);

#define DEFINE_INDEX_OPERATORS(OP)                               \
  IndexExpr operator OP(const IndexExpr& a, const IndexExpr& b); \
  IndexExpr operator OP(int64_t a, const IndexExpr& b);          \
  IndexExpr operator OP(const IndexExpr& a, int64_t b);          \
  IndexExpr operator OP(int32_t a, const IndexExpr& b);          \
  IndexExpr operator OP(const IndexExpr& a, int32_t b);

DEFINE_INDEX_OPERATORS(+)
DEFINE_INDEX_OPERATORS(-)
DEFINE_INDEX_OPERATORS(*)
DEFINE_INDEX_OPERATORS(/)
DEFINE_INDEX_OPERATORS(%)

#undef DEFINE_INDEX_OPERATORS

//-- left hand --
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator+(Expr a, POD b) {
  return Add::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator-(Expr a, POD b) {
  return Sub::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator*(Expr a, POD b) {
  return Mul::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator/(Expr a, POD b) {
  return Div::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator%(Expr a, POD b) {
  return Mod::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator<(Expr a, POD b) {
  return LT::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator<=(Expr a, POD b) {
  return LE::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator>(Expr a, POD b) {
  return GT::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator>=(Expr a, POD b) {
  return GE::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator==(Expr a, POD b) {
  return EQ::Make(Expr(a), Expr(b));
}

//- right hand --
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator+(POD a, Expr b) {
  return Add::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator-(POD a, Expr b) {
  return Sub::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator*(POD a, Expr b) {
  return Mul::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator/(POD a, Expr b) {
  return Div::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator%(POD a, Expr b) {
  return Mod::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator<(POD a, Expr b) {
  return LT::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator<=(POD a, Expr b) {
  return LE::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator>(POD a, Expr b) {
  return GT::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator>=(POD a, Expr b) {
  return GE::Make(Expr(a), Expr(b));
}
template <typename POD,
          typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator==(POD a, Expr b) {
  return EQ::Make(Expr(a), Expr(b));
}

//--
#define DEFINE_EXPR_OPERATOR(OP, FUNC)                         \
  inline Expr operator OP(const Expr& a, const Expr& b) {      \
    return FUNC::Make(a, b);                                   \
  }                                                            \
  inline Expr operator OP(const Expr& a, const IndexExpr& b) { \
    return FUNC::Make(a, Expr(b));                             \
  }                                                            \
  inline Expr operator OP(const IndexExpr& a, const Expr& b) { \
    return FUNC::Make(Expr(a), b);                             \
  }

DEFINE_EXPR_OPERATOR(+, Add)
DEFINE_EXPR_OPERATOR(-, Sub)
DEFINE_EXPR_OPERATOR(*, Mul)
DEFINE_EXPR_OPERATOR(/, Div)
DEFINE_EXPR_OPERATOR(%, Mod)

#undef DEFINE_EXPR_OPERATOR

inline Expr operator&&(Expr a, Expr b) { return And::Make(Expr(a), Expr(b)); }
inline Expr operator||(Expr a, Expr b) { return Or::Make(Expr(a), Expr(b)); }
inline Expr operator>=(Expr a, Expr b) { return GE::Make(Expr(a), Expr(b)); }
inline Expr operator<=(Expr a, Expr b) { return LE::Make(Expr(a), Expr(b)); }
inline Expr operator>(Expr a, Expr b) { return GT::Make(Expr(a), Expr(b)); }
inline Expr operator<(Expr a, Expr b) { return LT::Make(Expr(a), Expr(b)); }

inline Expr operator-(Expr a) {
  return (a.is_index()) ? Expr(-(a.as_index())).set_index(1)
                        : Minus::Make(Expr(a));
}
inline Expr operator!(Expr a) { return Not::Make(Expr(a)); }

Expr operator<<(Expr a, Expr b);
Expr operator>>(Expr a, Expr b);
Expr operator^(Expr a, Expr b);
Expr operator|(Expr a, Expr b);
Expr operator&(Expr a, Expr b);
Expr operator~(Expr a);

}  // namespace ir
}  // namespace cinn
