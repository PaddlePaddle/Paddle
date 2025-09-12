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

namespace ap::paddle {

struct DDimMethodClass {
  using This = DDimMethodClass;
  using Self = DDim;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    ss << "[";
    for (int i = 0; i < self.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << self.at(i);
    }
    ss << "]";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    int64_t hash_value = 0;
    for (int i = 0; i < self.size(); ++i) {
      hash_value = adt::hash_combine(hash_value, self.at(i));
    }
    return hash_value;
  }

  static adt::Result<axpr::Value> GetItem(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    const auto& index_val = args.at(0);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_LET_CONST_REF(index, index_val.template TryGet<int64_t>())
        << adt::errors::TypeError{std::string() +
                                  "'DDim.__getitem__()' takes integers, not " +
                                  axpr::GetTypeName(index_val) + "."};
    ADT_CHECK(index < self.size())
        << adt::errors::IndexError{"list index out of range"};
    return self.at(index);
  }
};

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetDDimClass() {
  using Impl = DDimMethodClass;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("DDim", [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__getitem__", &Impl::GetItem);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

}  // namespace ap::paddle
