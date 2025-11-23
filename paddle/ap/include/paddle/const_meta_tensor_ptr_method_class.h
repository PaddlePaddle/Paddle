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

#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/paddle/const_meta_tensor_ptr.h"
#include "paddle/ap/include/paddle/ddim_method_class.h"

namespace ap::paddle {

struct ConstMetaTensorPtrMethodClass {
  using This = ConstMetaTensorPtrMethodClass;
  using Self = ConstMetaTensorPtr;

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
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "dtype") {
      return This{}.GetDtype(self);
    }
    if (attr_name == "dims") {
      return This{}.GetDims(self);
    }
    return adt::errors::AttributeError{
        std::string() + "'ConstMetaTensorPtr' object has no attribute '" +
        attr_name + "'."};
  }

  adt::Result<axpr::Value> GetDims(const Self& self) {
    return GetDDimClass().New(self->dims());
  }

  adt::Result<axpr::Value> GetDtype(const Self& self) {
    ADT_LET_CONST_REF(dtype, axpr::GetDataTypeFromPhiDataType(self->dtype()));
    return dtype;
  }
};

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetConstMetaTensorPtrClass() {
  using Impl = ConstMetaTensorPtrMethodClass;
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "ConstMetaTensorPtr", [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__getattr__", &Impl::GetAttr);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

}  // namespace ap::paddle
