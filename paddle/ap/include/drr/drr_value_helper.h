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
#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/drr_value.h"

namespace ap::drr {

struct DrrValueHelper {
  using This = DrrValueHelper;

  DrrValue CastFromAxprValue(const axpr::Value& axpr_val) {
    return axpr_val.Match(
        [&](const axpr::BuiltinClassInstance<axpr::Value>& instance)
            -> DrrValue { return CastInstanceToDrrValue(instance); },
        [&](const auto&) -> DrrValue { return axpr_val; });
  }

  axpr::Value CastToAxprValue(const DrrValue& drr_value) {
    return drr_value.Match(
        [&](const axpr::Value& axpr_val) -> axpr::Value { return axpr_val; },
        [&](const auto& impl) -> axpr::Value {
          using T = std::decay_t<decltype(impl)>;
          using TT = drr::Type<T>;
          return TT::GetClass().New(impl);
        });
  }

 private:
  using AxprInstanceToDrrValueConverter =
      DrrValue (*)(const axpr::BuiltinClassInstance<axpr::Value>&);
  using AxprInstanceToDrrValueMap =
      std::map<std::type_index, AxprInstanceToDrrValueConverter>;

  DrrValue CastInstanceToDrrValue(
      const axpr::BuiltinClassInstance<axpr::Value>& instance) {
    const AxprInstanceToDrrValueMap& map = GetAxprInstanceToDrrValueMap();
    const auto& iter = map.find(instance.instance.type());
    if (iter == map.end()) {
      return axpr::Value{instance};
    } else {
      return iter->second(instance);
    }
  }

  const AxprInstanceToDrrValueMap& GetAxprInstanceToDrrValueMap() {
    static const AxprInstanceToDrrValueMap map(MakeAxprInstanceToDrrValueMap());
    return map;
  }

  AxprInstanceToDrrValueMap MakeAxprInstanceToDrrValueMap() {
    AxprInstanceToDrrValueMap map;
    InsertEntries(&map);
    return map;
  }

  template <int start_idx = 0>
  void InsertEntries(AxprInstanceToDrrValueMap* map) {
    if constexpr (start_idx >= std::variant_size_v<DrrValueImpl>) {
      (void)map;
      return;
    } else {
      using Impl = typename std::variant_alternative_t<start_idx, DrrValueImpl>;
      (*map)[typeid(Impl)] =
          &This::template ConvertAxprInstanceToDrrValue<Impl>;
      InsertEntries<start_idx + 1>(map);
    }
  }

  template <typename T>
  static DrrValue ConvertAxprInstanceToDrrValue(
      const axpr::BuiltinClassInstance<axpr::Value>& instance) {
    return std::any_cast<T>(instance.instance);
  }
};

}  // namespace ap::drr
