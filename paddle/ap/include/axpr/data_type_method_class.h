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
#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/ap/include/axpr/int_data_type.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/pointer_type_util.h"

namespace ap::axpr {

template <typename ValueT>
struct DataTypeMethodClass {
  using This = DataTypeMethodClass;
  using Self = DataType;

  adt::Result<ValueT> ToString(const Self& data_type) {
    return std::string("DataType.") + data_type.Name();
  }

  adt::Result<ValueT> Hash(const Self& data_type) {
    int64_t hash_value = std::hash<const char*>()("DataType");
    hash_value = adt::hash_combine(hash_value, data_type.index());
    return hash_value;
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template CastTo<std::string>());
    if (attr_name == "const_pointer_type") {
      return GetConstPointerType(self);
    }
    if (attr_name == "mutable_pointer_type") {
      return GetMutablePointerType(self);
    }
    return adt::errors::AttributeError{
        std::string() + "DataType has no attribute '" + attr_name + "'"};
  }

  adt::Result<ValueT> GetConstPointerType(const Self& self) {
    return GetConstPointerTypeFromDataType(self);
  }

  adt::Result<ValueT> GetMutablePointerType(const Self& self) {
    return GetMutablePointerTypeFromDataType(self);
  }

  template <typename BuiltinBinarySymbol>
  static BuiltinBinaryFunc<ValueT> GetBuiltinBinaryFunc() {
    if constexpr (std::is_same_v<BuiltinBinarySymbol, builtin_symbol::EQ>) {
      return &This::EQ;
    } else if constexpr (std::is_same_v<BuiltinBinarySymbol,  // NOLINT
                                        builtin_symbol::NE>) {
      return &This::NE;
    } else {
      return adt::Nothing{};
    }
  }

  static Result<ValueT> EQ(const ValueT& lhs_val, const ValueT& rhs_val) {
    ADT_LET_CONST_REF(lhs, lhs_val.template TryGet<DataType>());
    ADT_LET_CONST_REF(rhs, rhs_val.template TryGet<DataType>());
    const auto& pattern_match =
        ::common::Overloaded{[](auto lhs, auto rhs) -> ValueT {
          return std::is_same_v<decltype(lhs), decltype(rhs)>;
        }};
    return std::visit(pattern_match, lhs.variant(), rhs.variant());
  }

  static Result<ValueT> NE(const ValueT& lhs_val, const ValueT& rhs_val) {
    ADT_LET_CONST_REF(lhs, lhs_val.template TryGet<DataType>());
    ADT_LET_CONST_REF(rhs, rhs_val.template TryGet<DataType>());
    const auto& pattern_match =
        ::common::Overloaded{[](auto lhs, auto rhs) -> ValueT {
          return !std::is_same_v<decltype(lhs), decltype(rhs)>;
        }};
    return std::visit(pattern_match, lhs.variant(), rhs.variant());
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, DataType> : public DataTypeMethodClass<ValueT> {
};

template <typename ValueT>
struct TypeImplDataTypeMethodClass {
  using This = TypeImplDataTypeMethodClass;
  using Self = TypeImpl<DataType>;

  adt::Result<ValueT> GetAttr(const Self&, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, TryGetImpl<std::string>(attr_name_val));
    static const std::unordered_map<std::string, DataType> map{
#define MAKE_CPP_TYPE_CASE(cpp_type, enum_type)                              \
  {axpr::CppDataType<cpp_type>{}.Name(), DataType{CppDataType<cpp_type>{}}}, \
      {axpr::CppDataType<const cpp_type>{}.Name(),                           \
       DataType{                                                             \
           CppDataType<cpp_type>{}}},  // it's not a typo, DataType.const_int8
                                       // and DataType.int8 are treated
                                       // identical.

        PD_FOR_EACH_DATA_TYPE(MAKE_CPP_TYPE_CASE)
#undef MAKE_CPP_TYPE_CASE

#define MAKE_INT_CPP_TYPE_CASE(cpp_type)              \
  {#cpp_type, DataType{CppDataType<cpp_type##_t>{}}}, \
      {"const_" #cpp_type, DataType{CppDataType<cpp_type##_t>{}}},

            AP_FOR_EACH_INT_TYPE(MAKE_INT_CPP_TYPE_CASE)
#undef MAKE_INT_CPP_TYPE_CASE
                {"void", DataType{CppDataType<adt::Undefined>{}}},
    };
    const auto iter = map.find(attr_name);
    if (iter != map.end()) {
      return iter->second;
    }
    return adt::errors::AttributeError{
        std::string() + "class 'DataType' has no static attribute '" +
        attr_name + "'."};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<DataType>>
    : public TypeImplDataTypeMethodClass<ValueT> {};

}  // namespace ap::axpr
