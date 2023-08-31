// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_attribute.h"
#include "paddle/ir/core/builtin_attribute.h"

namespace ir {
namespace drr {

template <class T>
struct CppTypeToIrAttribute;

#define PD_SPECIALIZE_CppTypeToIrAttribute(cpp_type, ir_attr_type) \
  template <>                                                      \
  struct CppTypeToIrAttribute<cpp_type> {                          \
    using type = ir_attr_type;                                     \
  };

PD_SPECIALIZE_CppTypeToIrAttribute(bool, BoolAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(int32_t, Int32Attribute);
PD_SPECIALIZE_CppTypeToIrAttribute(int64_t, Int64Attribute);
PD_SPECIALIZE_CppTypeToIrAttribute(float, FloatAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(const std::string&, StrAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(phi::DataType,
                                   paddle::dialect::DataTypeAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(phi::Place, paddle::dialect::PlaceAttribute);

template <typename T>
struct IrAttrbuteCreator {
  typename CppTypeToIrAttribute<T>::type operator()(T obj) const {
    return CppTypeToIrAttribute<T>::type::template get(
        ir::IrContext::Instance(), obj);
  }
};

template <typename T>
struct IrAttrTypeCast {
  static T To(const ir::Attribute& attr) {
    return attr.dyn_cast<typename CppTypeToIrAttribute<T>::type>().data();
  }
};

template <>
struct IrAttrTypeCast<std::vector<int32_t>> {
  static std::vector<int32_t> To(const ir::Attribute& attr) {
    std::vector<int32_t> result;
    auto array_attr = attr.dyn_cast<ir::ArrayAttribute>();
    for (size_t i = 0; i < array_attr.size(); i++) {
      result.push_back(array_attr.at(i).dyn_cast<ir::Int32Attribute>().data());
    }
    return result;
  }
};

template <>
struct IrAttrTypeCast<std::vector<int64_t>> {
  static std::vector<int64_t> To(const ir::Attribute& attr) {
    std::vector<int64_t> result;
    if (attr.dyn_cast<ir::ArrayAttribute>()) {
      auto array_attr = attr.dyn_cast<ir::ArrayAttribute>();
      for (size_t i = 0; i < array_attr.size(); i++) {
        result.push_back(
            array_attr.at(i).dyn_cast<ir::Int64Attribute>().data());
      }
      return result;
    } else if (attr.dyn_cast<paddle::dialect::IntArrayAttribute>()) {
      result =
          attr.dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();
      return result;
    }
    PADDLE_THROW(phi::errors::Unavailable(
        "Dynamic cast failed for IR attribute vector<int64_t>"));
  }
};

}  // namespace drr
}  // namespace ir
