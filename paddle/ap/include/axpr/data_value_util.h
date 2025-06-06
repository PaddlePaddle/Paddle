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

#include "paddle/ap/include/axpr/binary_func.h"
#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/ap/include/axpr/data_value.h"
#include "paddle/ap/include/axpr/unary_func.h"

namespace ap::axpr {

namespace detail {

template <typename ArithmeticOp>
struct ArithmeticUnaryFuncHelper {
  static Result<DataValue> Call(const DataValue& value) {
    return value.Match([](auto val) -> Result<DataValue> {
      if constexpr (IsArithmeticOpSupported<decltype(val)>()) {
        return ArithmeticOp::Call(val);
      } else {
        return adt::errors::TypeError{
            std::string() + "unsupported operand type for " +
            ArithmeticOp::Name() + ": " + CppDataType<decltype(val)>{}.Name() +
            "."};
      }
    });
  }
};

template <typename ArithmeticOp>
struct ArithmeticBinaryOpHelper {
  template <typename LhsT, typename RhsT>
  static Result<DataValue> Call(LhsT lhs, RhsT rhs) {
    return ArithmeticOp::Call(lhs, rhs);
  }
};

template <>
struct ArithmeticBinaryOpHelper<ArithmeticDiv> {
  template <typename LhsT, typename RhsT>
  static Result<DataValue> Call(LhsT lhs, RhsT rhs) {
    if (rhs == 0) {
      return adt::errors::ZeroDivisionError{"division or modulo by zero"};
    }
    return ArithmeticDiv::Call(lhs, rhs);
  }
};

template <>
struct ArithmeticBinaryOpHelper<ArithmeticMod> {
  template <typename LhsT, typename RhsT>
  static Result<DataValue> Call(LhsT lhs, RhsT rhs) {
    if constexpr (std::is_integral_v<LhsT> && std::is_integral_v<RhsT>) {
      if (rhs == 0) {
        return adt::errors::ZeroDivisionError{"division or modulo by zero"};
      }
      return ArithmeticMod::Call(lhs, rhs);
    } else {
      return adt::errors::TypeError{
          std::string() + "'%' only support intergral type, but receive: '" +
          CppDataType<decltype(lhs)>{}.Name() + "' and '" +
          CppDataType<decltype(rhs)>{}.Name() + "'."};
    }
  }
};

template <typename ArithmeticOp>
struct ArithmeticBinaryFuncHelper {
  static Result<DataValue> Call(const DataValue& lhs_value,
                                const DataValue& rhs_value) {
    const auto& pattern_match =
        ::common::Overloaded{[](auto lhs, auto rhs) -> Result<DataValue> {
          if constexpr (IsArithmeticOpSupported<decltype(lhs)>() &&
                        IsArithmeticOpSupported<decltype(rhs)>()) {
            return ArithmeticBinaryOpHelper<ArithmeticOp>::Call(lhs, rhs);
          } else {
            return adt::errors::TypeError{
                std::string() + "unsupported operand types for " +
                ArithmeticOp::Name() + ": '" +
                CppDataType<decltype(lhs)>{}.Name() + "' and '" +
                CppDataType<decltype(rhs)>{}.Name() + "'."};
          }
        }};
    return std::visit(pattern_match, lhs_value.variant(), rhs_value.variant());
  }
};

}  // namespace detail

template <typename ArithmeticOp>
Result<DataValue> ArithmeticUnaryFunc(const DataValue& value) {
  return detail::ArithmeticUnaryFuncHelper<ArithmeticOp>::Call(value);
}

template <typename ArithmeticOp>
Result<DataValue> ArithmeticBinaryFunc(const DataValue& lhs_value,
                                       const DataValue& rhs_value) {
  return detail::ArithmeticBinaryFuncHelper<ArithmeticOp>::Call(lhs_value,
                                                                rhs_value);
}

}  // namespace ap::axpr
