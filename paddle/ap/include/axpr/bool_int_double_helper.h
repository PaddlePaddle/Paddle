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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/bool_int_double.h"
#include "paddle/ap/include/axpr/bool_int_double_arithmetic_util.h"

namespace ap::axpr {

template <typename ValueT>
struct BoolIntDoubleHelper {
  template <typename ArithmeticOp>
  static adt::Result<ValueT> BinaryFunc(const BoolIntDouble& lhs_val,
                                        const BoolIntDouble& rhs_val) {
    const auto& pattern_match = ::common::Overloaded{
        [&](const auto lhs, const auto rhs) -> adt::Result<ValueT> {
          return BoolIntDoubleArithmeticBinaryFunc<ArithmeticOp, ValueT>(lhs,
                                                                         rhs);
        }};
    return std::visit(pattern_match, lhs_val.variant(), rhs_val.variant());
  }
};

}  // namespace ap::axpr
