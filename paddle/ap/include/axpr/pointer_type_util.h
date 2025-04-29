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

#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/ap/include/axpr/pointer_type.h"

namespace ap::axpr {

PointerType RemoveConst(const PointerType& ptr_type);
PointerType GetConstPointerTypeFromDataType(const DataType& data_type);
PointerType GetMutablePointerTypeFromDataType(const DataType& data_type);
DataType GetDataTypeTypeFromPointerType(const PointerType& pointer_type);

namespace detail {

template <typename T>
struct TypeConverter;

#define SPECIALIZE_TYPE_CONVERTER(cpp_type, enum_type)    \
  template <>                                             \
  struct TypeConverter<CppPointerType<cpp_type*>> {       \
    using remove_const_type = CppPointerType<cpp_type*>;  \
  };                                                      \
  template <>                                             \
  struct TypeConverter<CppPointerType<const cpp_type*>> { \
    using remove_const_type = CppPointerType<cpp_type*>;  \
  };

PD_FOR_EACH_DATA_TYPE(SPECIALIZE_TYPE_CONVERTER);
#undef SPECIALIZE_TYPE_CONVERTER

template <>
struct TypeConverter<CppPointerType<void*>> {
  using remove_const_type = CppPointerType<void*>;
};

template <>
struct TypeConverter<CppPointerType<const void*>> {
  using remove_const_type = CppPointerType<void*>;
};

}  // namespace detail

inline PointerType RemoveConst(const PointerType& ptr_type) {
  return ptr_type.Match([](auto impl) {
    return PointerType{
        typename detail::TypeConverter<decltype(impl)>::remove_const_type{}};
  });
}

inline PointerType GetConstPointerTypeFromDataType(const DataType& data_type) {
  return data_type.Match([&](const auto& impl) -> PointerType {
    using T = typename std::decay_t<decltype(impl)>::type;
    if constexpr (std::is_same_v<T, adt::Undefined>) {
      return CppPointerType<const void*>{};
    } else {
      return CppPointerType<const T*>{};
    }
  });
}

inline PointerType GetMutablePointerTypeFromDataType(
    const DataType& data_type) {
  return data_type.Match([&](const auto& impl) -> PointerType {
    using T = typename std::decay_t<decltype(impl)>::type;
    if constexpr (std::is_same_v<T, adt::Undefined>) {
      return CppPointerType<void*>{};
    } else {
      return CppPointerType<T*>{};
    }
  });
}

inline DataType GetDataTypeTypeFromPointerType(
    const PointerType& pointer_type) {
  return pointer_type.Match(
      [&](CppPointerType<const void*>) -> DataType {
        return CppDataType<adt::Undefined>{};
      },
      [&](CppPointerType<void*>) -> DataType {
        return CppDataType<adt::Undefined>{};
      },
      [&](const auto& impl) -> DataType {
        using PtrT = typename std::decay_t<decltype(impl)>::type;
        using T = std::remove_const_t<std::remove_pointer_t<PtrT>>;
        return CppDataType<T>{};
      });
}
}  // namespace ap::axpr
