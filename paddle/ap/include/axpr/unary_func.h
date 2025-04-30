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

namespace ap::axpr {

#define PEXPR_FOR_EACH_UNARY_OP(_) \
  _(Not, !)                        \
  _(Neg, -)

#define DEFINE_ARITHMETIC_UNARY_OP(name, op)            \
  struct Arithmetic##name {                             \
    static constexpr const char* Name() { return #op; } \
                                                        \
    template <typename LhsT>                            \
    static auto Call(const LhsT& val) {                 \
      return op val;                                    \
    }                                                   \
  };
PEXPR_FOR_EACH_UNARY_OP(DEFINE_ARITHMETIC_UNARY_OP);
#undef DEFINE_ARITHMETIC_UNARY_OP

template <typename ArithmeticOp>
struct BoolIntDoubleUnary {
  static constexpr const char* Name() { return ArithmeticOp::Name(); }
  template <typename T>
  static auto Call(T operand) {
    auto ret = ArithmeticOp::Call(operand);
    using RetT = decltype(ret);
    if constexpr (std::is_same_v<RetT, bool>) {
      return ret;
    } else if constexpr (std::is_integral_v<RetT>) {
      return static_cast<int64_t>(ret);
    } else {
      static_assert(std::is_floating_point<RetT>::value, "");
      return static_cast<double>(ret);
    }
  }
};

}  // namespace ap::axpr
