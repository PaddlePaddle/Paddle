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
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/ap/include/axpr/pointer_type.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

using PointerValueImpl = std::variant<
#define MAKE_ARG_VALUE_ALTERNATIVE(cpp_type, enum_type) \
  cpp_type*, const cpp_type*,
    PD_FOR_EACH_DATA_TYPE(MAKE_ARG_VALUE_ALTERNATIVE) void*,
    const void*
#undef MAKE_ARG_VALUE_ALTERNATIVE
    >;

struct PointerValue : public PointerValueImpl {
  using PointerValueImpl::PointerValueImpl;
  ADT_DEFINE_VARIANT_METHODS(PointerValueImpl);

  PointerType GetType() const {
    return Match([](auto impl) -> PointerType {
      return PointerType{CppPointerType<decltype(impl)>{}};
    });
  }
};

template <>
struct TypeImpl<PointerValue> : public std::monostate {
  using value_type = PointerValue;

  const char* Name() const { return "PointerValue"; }
};

}  // namespace ap::axpr
