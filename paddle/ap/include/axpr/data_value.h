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
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

using DataValueImpl = std::variant<
#define MAKE_ARG_VALUE_ALTERNATIVE(cpp_type, enum_type) cpp_type,
    PD_FOR_EACH_DATA_TYPE(MAKE_ARG_VALUE_ALTERNATIVE) adt::Undefined
#undef MAKE_ARG_VALUE_ALTERNATIVE
    >;

struct DataValue : public DataValueImpl {
  using DataValueImpl::DataValueImpl;
  ADT_DEFINE_VARIANT_METHODS(DataValueImpl);

  DataType GetType() const {
    return Match(
        [](auto impl) -> DataType { return CppDataType<decltype(impl)>{}; });
  }

  Result<DataValue> StaticCastTo(const DataType& dst_type) const {
    const auto& pattern_match = ::common::Overloaded{
        [&](auto arg_type_impl, auto cpp_value_impl) -> Result<DataValue> {
          using DstT = typename decltype(arg_type_impl)::type;
          return DataValueStaticCast<DstT>(cpp_value_impl);
        }};
    return std::visit(pattern_match, dst_type.variant(), this->variant());
  }

  Result<std::string> ToString() const {
    return Match([](const auto& impl) -> adt::Result<std::string> {
      using T = std::decay_t<decltype(impl)>;
      if constexpr (std::is_same_v<T, bool>) {
        return std::to_string(impl);
      } else if constexpr (std::is_integral_v<T>) {
        return std::to_string(impl);
      } else if constexpr (std::is_same_v<T, float>) {
        return std::to_string(impl);
      } else if constexpr (std::is_same_v<T, double>) {
        return std::to_string(impl);
      } else {
        return adt::errors::NotImplementedError{"DataType NotImplemented."};
      }
    });
  }

  Result<int64_t> GetHashValue() const {
    using RetT = Result<int64_t>;
    return Match([](const auto& impl) -> RetT {
      using T = std::decay_t<decltype(impl)>;
      if constexpr (std::is_same_v<T, bool>) {
        return static_cast<int64_t>(std::hash<T>()(impl));
      } else if constexpr (std::is_integral_v<T>) {
        return static_cast<int64_t>(std::hash<T>()(impl));
      } else if constexpr (std::is_same_v<T, float>) {
        return static_cast<int64_t>(std::hash<T>()(impl));
      } else if constexpr (std::is_same_v<T, double>) {
        return static_cast<int64_t>(std::hash<T>()(impl));
      } else {
        return adt::errors::NotImplementedError{"DataType NotImplemented."};
      }
    });
  }

 private:
  template <typename DstT, typename SrcT>
  Result<DataValue> DataValueStaticCast(SrcT v) const {
    if constexpr (std::is_same_v<DstT, adt::Undefined>) {
      return adt::errors::TypeError{
          "static_cast can not cast to 'undefined' type."};
    } else if constexpr (std::is_same_v<DstT, phi::dtype::pstring>) {
      return adt::errors::TypeError{
          "static_cast can not cast to 'pstring' type."};
    } else if constexpr (std::is_same_v<SrcT, adt::Undefined>) {
      return adt::errors::TypeError{
          "static_cast can not cast from 'undefined' type."};
    } else if constexpr (std::is_same_v<SrcT, phi::dtype::pstring>) {
      return adt::errors::TypeError{
          "static_cast can not cast from 'pstring' type."};
    } else {
      return static_cast<DstT>(v);
    }
  }
};

template <>
struct TypeImpl<DataValue> : public std::monostate {
  using value_type = DataValue;

  const char* Name() const { return "DataValue"; }
};

}  // namespace ap::axpr
