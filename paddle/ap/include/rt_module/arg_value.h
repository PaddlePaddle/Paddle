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
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/code_module/arg_type.h"
#include "paddle/ap/include/code_module/data_type.h"

namespace ap::rt_module {

using code_module::ArgType;

using ArgValueImpl = std::variant<ap::axpr::DataValue, ap::axpr::PointerValue>;

struct ArgValue : public ArgValueImpl {
  using ArgValueImpl::ArgValueImpl;
  ADT_DEFINE_VARIANT_METHODS(ArgValueImpl);

  ArgType GetType() const {
    return Match([](auto impl) -> ArgType { return impl.GetType(); });
  }

  template <typename ValueT>
  adt::Result<ValueT> CastTo() const {
    return Match([](const auto& impl) -> adt::Result<ValueT> { return impl; });
  }

  template <typename T>
  adt::Result<T> TryGetValue() const {
    if constexpr (std::is_pointer_v<T>) {
      const auto& pointer_value =
          this->template TryGet<ap::axpr::PointerValue>();
      ADT_RETURN_IF_ERR(pointer_value);
      return pointer_value.GetOkValue().template TryGet<T>();
    } else {
      const auto& data_value = this->template TryGet<ap::axpr::DataValue>();
      ADT_RETURN_IF_ERR(data_value);
      return data_value.GetOkValue().template TryGet<T>();
    }
  }
};

template <typename ValueT>
Result<ArgValue> CastToArgValue(const ValueT& value) {
  return value.Match(
      [&](const ap::axpr::DataValue& impl) -> Result<ArgValue> { return impl; },
      [&](const ap::axpr::PointerValue& impl) -> Result<ArgValue> {
        return impl;
      },
      [&](const auto&) -> Result<ArgValue> {
        return TypeError{std::string() +
                         "CastToArgValue failed. expected types: "
                         "(DataValue, PointerValue), actual type: " +
                         axpr::GetTypeName(value)};
      });
}

}  // namespace ap::rt_module
