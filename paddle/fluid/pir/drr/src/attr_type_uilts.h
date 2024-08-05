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

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace paddle {
namespace drr {

template <class T>
struct CppTypeToIrAttribute;

#define PD_SPECIALIZE_CppTypeToIrAttribute(cpp_type, ir_attr_type) \
  template <>                                                      \
  struct CppTypeToIrAttribute<                                     \
      std::remove_const_t<std::remove_reference_t<cpp_type>>> {    \
    using type = ir_attr_type;                                     \
  };

PD_SPECIALIZE_CppTypeToIrAttribute(bool, pir::BoolAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(int32_t, pir::Int32Attribute);
PD_SPECIALIZE_CppTypeToIrAttribute(int64_t, pir::Int64Attribute);
PD_SPECIALIZE_CppTypeToIrAttribute(float, pir::FloatAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(double, pir::DoubleAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(std::string, pir::StrAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(std::vector<int32_t>, pir::ArrayAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(std::vector<int64_t>,
                                   paddle::dialect::IntArrayAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(std::vector<float>, pir::ArrayAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(phi::DataType,
                                   paddle::dialect::DataTypeAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(phi::Place, paddle::dialect::PlaceAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(phi::DataLayout,
                                   paddle::dialect::DataLayoutAttribute);
PD_SPECIALIZE_CppTypeToIrAttribute(phi::IntArray,
                                   paddle::dialect::IntArrayAttribute);

template <typename T>
struct IrAttributeCreator {
  typename CppTypeToIrAttribute<T>::type operator()(T obj) const {
    return CppTypeToIrAttribute<T>::type::template get(
        pir::IrContext::Instance(), obj);
  }
};

template <>
struct IrAttributeCreator<std::vector<int32_t>> {
  pir::ArrayAttribute operator()(std::vector<int32_t> obj) const {
    std::vector<pir::Attribute> attr_vec;
    attr_vec.reserve(obj.size());
    for (int32_t x : obj) {
      attr_vec.push_back(
          pir::Int32Attribute::get(pir::IrContext::Instance(), x));
    }
    return pir::ArrayAttribute::get(pir::IrContext::Instance(), attr_vec);
  }
};

template <>
struct IrAttributeCreator<std::vector<float>> {
  pir::ArrayAttribute operator()(std::vector<float> obj) const {
    std::vector<pir::Attribute> attr_vec;
    attr_vec.reserve(obj.size());
    for (float x : obj) {
      attr_vec.push_back(
          pir::FloatAttribute::get(pir::IrContext::Instance(), x));
    }
    return pir::ArrayAttribute::get(pir::IrContext::Instance(), attr_vec);
  }
};

template <typename T>
struct IrAttrTypeCast {
  static T To(const pir::Attribute& attr) {
    return attr.dyn_cast<typename CppTypeToIrAttribute<T>::type>().data();
  }
};

template <>
struct IrAttrTypeCast<std::string> {
  static std::string To(const pir::Attribute& attr) {
    return attr.dyn_cast<typename CppTypeToIrAttribute<std::string>::type>()
        .AsString();
  }
};

template <>
struct IrAttrTypeCast<std::vector<int32_t>> {
  static std::vector<int32_t> To(const pir::Attribute& attr) {
    std::vector<int32_t> result;
    auto array_attr = attr.dyn_cast<pir::ArrayAttribute>();
    for (size_t i = 0; i < array_attr.size(); i++) {
      result.push_back(array_attr.at(i).dyn_cast<pir::Int32Attribute>().data());
    }
    return result;
  }
};

template <>
struct IrAttrTypeCast<std::vector<int64_t>> {
  static std::vector<int64_t> To(const pir::Attribute& attr) {
    std::vector<int64_t> result;
    if (attr.isa<pir::ArrayAttribute>()) {
      auto array_attr = attr.dyn_cast<pir::ArrayAttribute>();
      for (size_t i = 0; i < array_attr.size(); i++) {
        result.push_back(
            array_attr.at(i).dyn_cast<pir::Int64Attribute>().data());
      }
    } else if (attr.isa<paddle::dialect::IntArrayAttribute>()) {
      result =
          attr.dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "Dynamic cast failed for IR attribute vector<int64_t>"));
    }
    return result;
  }
};

template <>
struct IrAttrTypeCast<std::vector<float>> {
  static std::vector<float> To(const pir::Attribute& attr) {
    std::vector<float> result;
    auto array_attr = attr.dyn_cast<pir::ArrayAttribute>();
    for (size_t i = 0; i < array_attr.size(); i++) {
      result.push_back(array_attr.at(i).dyn_cast<pir::FloatAttribute>().data());
    }
    return result;
  }
};

}  // namespace drr
}  // namespace paddle
