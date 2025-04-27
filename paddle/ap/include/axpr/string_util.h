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

#include <string>
#include "paddle/ap/include/axpr/constants.h"
#include "paddle/ap/include/axpr/method_class.h"

namespace ap::axpr {

template <typename ArithmeticOp, typename Val, typename T>
struct BuiltinStringBinaryHelper {
  static Result<Val> Call(const std::string& str, const T& rhs) {
    return adt::errors::TypeError{
        std::string() + "unsupported operand types for " +
        ArithmeticOp::Name() + ": 'str' and '" + TypeImpl<T>{}.Name() + "'"};
  }
};

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticAdd, Val, std::string> {
  static Result<Val> Call(const std::string& lhs, const std::string& rhs) {
    return ArithmeticAdd::Call(lhs, rhs);
  }
};

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticAdd, Val, int64_t> {
  static Result<Val> Call(const std::string& lhs, const int64_t& rhs_val) {
    return lhs + std::to_string(rhs_val);
  }
};

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticAdd, Val, bool> {
  static Result<Val> Call(const std::string& lhs, const bool& rhs_val) {
    return lhs + (rhs_val ? "True" : "False");
  }
};

#define SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(cls_name)             \
  template <typename Val>                                                     \
  struct BuiltinStringBinaryHelper<cls_name, Val, std::string> {              \
    static Result<Val> Call(const std::string& lhs, const std::string& rhs) { \
      return cls_name::Call(lhs, rhs);                                        \
    }                                                                         \
  };
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticEQ);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticNE);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticGT);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticGE);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticLT);
SPECIALIZE_BuiltinStringBinaryHelper_string_cmp(ArithmeticLE);
#undef SPECIALIZE_BuiltinStringBinaryHelper_string

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticMul, Val, int64_t> {
  static Result<Val> Call(const std::string& lhs, int64_t size) {
    size = (size > 0 ? size : 0);
    std::ostringstream ss;
    for (int i = 0; i < size; ++i) {
      ss << lhs;
    }
    return ss.str();
  }
};

template <typename Val>
struct BuiltinStringBinaryHelper<ArithmeticMul, Val, bool> {
  static Result<Val> Call(const std::string& lhs, bool size) {
    std::ostringstream ss;
    for (int i = 0; i < static_cast<int>(size); ++i) {
      ss << lhs;
    }
    return ss.str();
  }
};

template <typename ArithmeticOp, typename Val>
Result<Val> BuiltinStringBinary(const std::string& str, const Val& rhs_val) {
  return rhs_val.Match(
      [&](const BuiltinClassInstance<Val>& impl) -> Result<Val> {
        return adt::errors::TypeError{
            std::string() + "unsupported operand types for " +
            ArithmeticOp::Name() + ": 'str' and '" +
            impl.type.class_attrs()->class_name + "'"};
      },
      [&](const ClassInstance<Val>& impl) -> Result<Val> {
        return adt::errors::TypeError{std::string() +
                                      "unsupported operand types for " +
                                      ArithmeticOp::Name() + ": 'str' and '" +
                                      impl->type.class_attrs->class_name + "'"};
      },
      [&](const auto& rhs) -> Result<Val> {
        using T = std::decay_t<decltype(rhs)>;
        return BuiltinStringBinaryHelper<ArithmeticOp, Val, T>::Call(str, rhs);
      });
}

}  // namespace ap::axpr
