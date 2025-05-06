// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <type_traits>

namespace ap::axpr {

#define PEXPR_FOR_EACH_BINARY_OP(_) \
  _(Add, +)                         \
  _(Sub, -)                         \
  _(Mul, *)                         \
  _(Div, /)                         \
  _(Mod, %)                         \
  _(EQ, ==)                         \
  _(NE, !=)                         \
  _(GT, >)                          \
  _(GE, >=)                         \
  _(LT, <)                          \
  _(LE, <=)

#define DEFINE_ARITHMETIC_BINARY_OP(name, op)            \
  struct Arithmetic##name {                              \
    static constexpr const char* Name() { return #op; }  \
                                                         \
    template <typename LhsT, typename RhsT>              \
    static auto Call(const LhsT& lhs, const RhsT& rhs) { \
      using T = std::common_type_t<LhsT, RhsT>;          \
      return static_cast<T>(lhs) op static_cast<T>(rhs); \
    }                                                    \
  };
PEXPR_FOR_EACH_BINARY_OP(DEFINE_ARITHMETIC_BINARY_OP);
#undef DEFINE_ARITHMETIC_BINARY_OP

template <typename ArithmeticOp>
struct BoolIntDoubleBinary {
  static constexpr const char* Name() { return ArithmeticOp::Name(); }
  template <typename LhsT, typename RhsT>
  static auto Call(LhsT lhs, RhsT rhs) {
    auto ret = ArithmeticOp::Call(lhs, rhs);
    using T = decltype(ret);
    if constexpr (std::is_same_v<T, bool>) {
      return ret;
    } else if constexpr (std::is_integral_v<T>) {
      return static_cast<int64_t>(ret);
    } else {
      static_assert(std::is_floating_point<T>::value, "");
      return static_cast<double>(ret);
    }
  }
};

}  // namespace ap::axpr
