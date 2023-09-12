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
inline Expr operator+(Expr a, Expr b) { return Add::Make(a, b); }
inline Expr operator-(Expr a, Expr b) { return Sub::Make(a, b); }
inline Expr operator*(Expr a, Expr b) { return Mul::Make(a, b); }
inline Expr operator/(Expr a, Expr b) { return Div::Make(a, b); }
inline Expr operator%(Expr a, Expr b) { return Mod::Make(a, b); }

inline Expr operator&&(Expr a, Expr b) { return And::Make(Expr(a), Expr(b)); }
inline Expr operator||(Expr a, Expr b) { return Or::Make(Expr(a), Expr(b)); }
inline Expr operator>=(Expr a, Expr b) { return GE::Make(Expr(a), Expr(b)); }
inline Expr operator<=(Expr a, Expr b) { return LE::Make(Expr(a), Expr(b)); }
inline Expr operator>(Expr a, Expr b) { return GT::Make(Expr(a), Expr(b)); }
inline Expr operator<(Expr a, Expr b) { return LT::Make(Expr(a), Expr(b)); }

inline Expr operator-(Expr a) { return Minus::Make(Expr(a)); }
inline Expr operator!(Expr a) { return Not::Make(Expr(a)); }

Expr operator<<(Expr a, Expr b);
Expr operator>>(Expr a, Expr b);
Expr operator^(Expr a, Expr b);
Expr operator|(Expr a, Expr b);
Expr operator&(Expr a, Expr b);
Expr operator~(Expr a);

}  // namespace ir
}  // namespace cinn
