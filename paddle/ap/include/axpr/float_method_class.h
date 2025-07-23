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

#include <cstdint>
#include "paddle/ap/include/axpr/bool_int_double_arithmetic_util.h"
#include "paddle/ap/include/axpr/constants.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
struct FloatMethodClass {
  using This = FloatMethodClass;
  using Self = double;

  adt::Result<ValueT> ToString(const Self val) { return std::to_string(val); }

  adt::Result<ValueT> Hash(const Self val) {
    return static_cast<int64_t>(std::hash<Self>()(val));
  }

  template <typename BuiltinUnarySymbol>
  static BuiltinUnaryFunc<ValueT> GetBuiltinUnaryFunc() {
    if constexpr (ConvertBuiltinSymbolToArithmetic<
                      BuiltinUnarySymbol>::convertible) {
      using ArithmeticOp = typename ConvertBuiltinSymbolToArithmetic<
          BuiltinUnarySymbol>::arithmetic_op_type;
      return &This::UnaryFunc<ArithmeticOp>;
    } else {
      return adt::Nothing{};
    }
  }

  template <typename BuiltinBinarySymbol>
  static BuiltinBinaryFunc<ValueT> GetBuiltinBinaryFunc() {
    if constexpr (ConvertBuiltinSymbolToArithmetic<
                      BuiltinBinarySymbol>::convertible) {
      using ArithmeticOp = typename ConvertBuiltinSymbolToArithmetic<
          BuiltinBinarySymbol>::arithmetic_op_type;
      return &This::template BinaryFunc<ArithmeticOp>;
    } else {
      return adt::Nothing{};
    }
  }

  template <typename ArithmeticOp>
  static adt::Result<ValueT> BinaryFunc(const ValueT& lhs_val,
                                        const ValueT& rhs_val) {
    ADT_LET_CONST_REF(lhs, lhs_val.template TryGet<double>());
    return rhs_val.Match(
        [&](bool rhs) -> adt::Result<ValueT> {
          return BoolIntDoubleArithmeticBinaryFunc<ArithmeticOp, ValueT>(lhs,
                                                                         rhs);
        },
        [&](int64_t rhs) -> adt::Result<ValueT> {
          return BoolIntDoubleArithmeticBinaryFunc<ArithmeticOp, ValueT>(lhs,
                                                                         rhs);
        },
        [&](double rhs) -> adt::Result<ValueT> {
          return BoolIntDoubleArithmeticBinaryFunc<ArithmeticOp, ValueT>(lhs,
                                                                         rhs);
        },
        [&](const auto& impl) -> adt::Result<ValueT> {
          return adt::errors::TypeError{std::string() +
                                        "unsupported operand type(s) for " +
                                        ArithmeticOp::Name() + ": 'int' and '" +
                                        axpr::GetTypeName(rhs_val) + "'"};
        });
  }

  template <typename ArithmeticOp>
  static adt::Result<ValueT> UnaryFunc(const ValueT& val) {
    ADT_LET_CONST_REF(operand, val.template TryGet<double>());
    return BoolIntDoubleArithmeticUnaryFunc<ArithmeticOp, ValueT>(operand);
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, double> : public FloatMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<double>> {
  using This = MethodClassImpl<ValueT, TypeImpl<double>>;

  adt::Result<ValueT> Call(const TypeImpl<double>&) { return &This::Construct; }

  static adt::Result<ValueT> Construct(const ValueT&,
                                       const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "float() takes 1 argument, but " +
        std::to_string(args.size()) + " were given"};
    using T = double;
    using RetT = adt::Result<ValueT>;
    return args.at(0).Match(
        [&](bool c) -> RetT { return static_cast<T>(c); },
        [&](int64_t c) -> RetT { return static_cast<T>(c); },
        [&](double c) -> RetT { return static_cast<T>(c); },
        [&](DataValue data_value) -> RetT {
          return data_value.Match(
              [&](const axpr::pstring&) -> RetT {
                return adt::errors::TypeError{
                    "invalid conversion from type 'pstring' to 'float'"};
              },
              [&](const adt::Undefined&) -> RetT {
                return adt::errors::TypeError{
                    "invalid conversion from type 'void' to 'float'"};
              },
              [&](const auto& impl) -> RetT { return static_cast<T>(impl); });
        },
        [&](const auto&) -> adt::Result<ValueT> {
          return adt::errors::TypeError{
              std::string() +
              "the argument 1 of float() should be bool/int/float/DataValue"};
        });
  }
};

}  // namespace ap::axpr
