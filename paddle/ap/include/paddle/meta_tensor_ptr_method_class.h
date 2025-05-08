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

#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/paddle/ddim.h"
#include "paddle/ap/include/paddle/ddim_method_class.h"
#include "paddle/ap/include/paddle/meta_tensor_ptr.h"

namespace ap::paddle {

struct MetaTensorPtrMethodClass {
  using This = MetaTensorPtrMethodClass;
  using Self = MetaTensorPtr;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const auto* ptr = self;
    ss << "<" << axpr::TypeImpl<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return reinterpret_cast<int64_t>(self);
  }

  static adt::Result<axpr::Value> GetAttr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(attr_name, attr_name_val.template CastTo<std::string>());
    if (attr_name == "dtype") {
      return This{}.GetDtype(self);
    }
    if (attr_name == "dims") {
      return This{}.GetDims(self);
    }
    return adt::errors::AttributeError{
        std::string() + "'MetaTensorPtr' object has no attribute '" +
        attr_name + "'."};
  }

  adt::Result<axpr::Value> GetDims(const Self& self) {
    return GetDDimClass().New(self->dims());
  }

  adt::Result<axpr::Value> GetDtype(const Self& self) {
    ADT_LET_CONST_REF(dtype, axpr::GetDataTypeFromPhiDataType(self->dtype()));
    return dtype;
  }

  static adt::Result<axpr::Value> SetAttr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 2);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(attr_name, attr_name_val.template CastTo<std::string>());
    if (attr_name == "dtype") {
      return StaticSetDtype(self_val, args);
    }
    if (attr_name == "dims") {
      return StaticSetDims(self_val, args);
    }
    return adt::errors::AttributeError{
        std::string() + "'MetaTensorPtr' object has no attribute '" +
        attr_name + "'."};
  }

  static adt::Result<axpr::Value> StaticSetDtype(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(data_type, args.at(1).template CastTo<axpr::DataType>());
    ADT_LET_CONST_REF(dtype, GetPhiDataTypeFromDataType(data_type));
    self->set_dtype(dtype);
    return adt::Nothing{};
  }

  static adt::Result<axpr::Value> StaticSetDims(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 2);
    return This{}.SetDims(self, args.at(1));
  }

  adt::Result<axpr::Value> SetDims(const Self& self,
                                   const axpr::Value& dims_val) {
    return dims_val.Match(
        [&](const DDim& ddims) -> adt::Result<axpr::Value> {
          return SetDimsByDDim(self, ddims);
        },
        [&](const adt::List<axpr::Value>& list) -> adt::Result<axpr::Value> {
          return SetDimsByIntList(self, list);
        },
        [&](const auto&) -> adt::Result<axpr::Value> {
          return adt::errors::TypeError{"only DDim or list of int supported."};
        });
  }

  adt::Result<axpr::Value> SetDimsByDDim(const Self& self, const DDim& ddims) {
    self->set_dims(ddims);
    return adt::Nothing{};
  }

  adt::Result<axpr::Value> SetDimsByIntList(
      const Self& self, const adt::List<axpr::Value>& list) {
    std::vector<int64_t> dims{};
    dims.reserve(list->size());
    for (const auto& dim_val : *list) {
      ADT_LET_CONST_REF(dim, dim_val.template CastTo<int64_t>());
      dims.push_back(dim);
    }
    self->set_dims(::common::make_ddim(dims));
    return adt::Nothing{};
  }
};

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetMetaTensorPtrClass() {
  using Impl = MetaTensorPtrMethodClass;
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "MetaTensorPtr", [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__getattr__", &Impl::GetAttr);
        Define("__setattr__", &Impl::SetAttr);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

}  // namespace ap::paddle
