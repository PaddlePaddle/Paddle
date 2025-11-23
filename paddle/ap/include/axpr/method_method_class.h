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

#include "paddle/ap/include/axpr/constants.h"
#include "paddle/ap/include/axpr/method.h"
#include "paddle/ap/include/axpr/method_class.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodMethodClass {
  using Self = MethodMethodClass;

  template <typename BuiltinUnarySymbol>
  static BuiltinUnaryFunc<ValueT> GetBuiltinUnaryFunc() {
    return adt::Nothing{};
  }

  template <typename BuiltinBinarySymbol>
  static BuiltinBinaryFunc<ValueT> GetBuiltinBinaryFunc() {
    return adt::Nothing{};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, Method<ValueT>> {
  using method_class = MethodMethodClass<ValueT>;

  template <typename BuiltinUnarySymbol>
  static BuiltinUnaryFunc<ValueT> GetBuiltinUnaryFunc() {
    return method_class::template GetBuiltinUnaryFunc<BuiltinUnarySymbol>();
  }

  template <typename BuiltinBinarySymbol>
  static BuiltinBinaryFunc<ValueT> GetBuiltinBinaryFunc() {
    return method_class::template GetBuiltinBinaryFunc<BuiltinBinarySymbol>();
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<Method<ValueT>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
