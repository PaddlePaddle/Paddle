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
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/paddle/const_meta_tensor_ptr_method_class.h"

namespace ap::paddle {

struct ConstStdVectorConstMetaTensorPtrPtrMethodClass {
  using This = ConstStdVectorConstMetaTensorPtrPtrMethodClass;
  using Self = const std::vector<ConstMetaTensorPtr>*;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const void* ptr = self;
    ss << "<ConstStdVectorConstMetaTensorPtrPtr object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return reinterpret_cast<int64_t>(self);
  }

  static adt::Result<axpr::Value> GetItem(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    const auto& idx_val = args.at(0);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_LET_CONST_REF(idx, idx_val.template CastTo<int64_t>())
        << adt::errors::TypeError{std::string() +
                                  "vector indices must be integers, not " +
                                  axpr::GetTypeName(idx_val)};
    int64_t index = idx;
    if (index < 0) {
      index += self->size();
    }
    if (index >= 0 && index < static_cast<int64_t>(self->size())) {
      return CastItem(self->at(index));
    }
    return adt::errors::IndexError{"vector index out of range"};
  }

  static adt::Result<axpr::Value> CastItem(const ConstMetaTensorPtr& elem) {
    return GetConstMetaTensorPtrClass().New(elem);
  }
};

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetConstStdVectorConstMetaTensorPtrPtrClass() {
  using Impl = ConstStdVectorConstMetaTensorPtrPtrMethodClass;
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "ConstStdVectorConstMetaTensorPtrPtr", [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__getitem__", &Impl::GetItem);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

}  // namespace ap::paddle
