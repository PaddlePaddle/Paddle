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
#include "paddle/ap/include/axpr/method_class.h"

namespace ap::axpr {

template <typename ValueT>
struct NothingMethodClass {
  using This = NothingMethodClass;
  using Self = adt::Nothing;

  adt::Result<ValueT> ToString(const Self&) { return std::string(""); }

  adt::Result<ValueT> Hash(const Self& self) { return static_cast<int64_t>(0); }

  Result<ValueT> EQ(const ValueT& lhs_val, const ValueT& rhs_val) {
    const auto& opt_lhs = lhs_val.template TryGet<adt::Nothing>();
    ADT_RETURN_IF_ERR(opt_lhs);
    return rhs_val.Match([](adt::Nothing) -> ValueT { return true; },
                         [](const auto&) -> ValueT { return false; });
  }

  Result<ValueT> NE(const ValueT& lhs_val, const ValueT& rhs_val) {
    const auto& opt_lhs = lhs_val.template TryGet<adt::Nothing>();
    ADT_RETURN_IF_ERR(opt_lhs);
    return rhs_val.Match([](adt::Nothing) -> ValueT { return false; },
                         [](const auto&) -> ValueT { return true; });
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, adt::Nothing>
    : public NothingMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<adt::Nothing>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
