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
#include "paddle/ap/include/paddle/pir_node.h"

namespace ap::paddle {

struct PirNodeHelper {
  using This = PirNodeHelper;

  adt::Result<PirNode> CastFromAxprValue(const axpr::Value& axpr_val) {
    using RetT = adt::Result<PirNode>;
    return axpr_val.Match(
        [&](const axpr::BuiltinClassInstance<axpr::Value>& instance) -> RetT {
          return CastInstanceToPirNode(instance);
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              std::string() + "PirNodeHelper::CastFromAxprValue() failed"};
        });
  }

 private:
  using AxprInstanceToPirNodeConverter =
      adt::Result<PirNode> (*)(const axpr::BuiltinClassInstance<axpr::Value>&);
  using AxprInstanceToPirNodeMap =
      std::map<std::type_index, AxprInstanceToPirNodeConverter>;

  adt::Result<PirNode> CastInstanceToPirNode(
      const axpr::BuiltinClassInstance<axpr::Value>& instance) {
    const AxprInstanceToPirNodeMap& map = GetAxprInstanceToPirNodeMap();
    const auto& iter = map.find(instance.instance.type());
    if (iter == map.end()) {
      return adt::errors::TypeError{
          "PirNodeHelper::CastInstanceToPirNode failed"};
    } else {
      return iter->second(instance);
    }
  }

  const AxprInstanceToPirNodeMap& GetAxprInstanceToPirNodeMap() {
    static const AxprInstanceToPirNodeMap map(MakeAxprInstanceToPirNodeMap());
    return map;
  }

  AxprInstanceToPirNodeMap MakeAxprInstanceToPirNodeMap() {
    AxprInstanceToPirNodeMap map;
    InsertEntries(&map);
    return map;
  }

  template <int start_idx = 0>
  void InsertEntries(AxprInstanceToPirNodeMap* map) {
    if constexpr (start_idx >= std::variant_size_v<PirNodeImpl>) {
      return;
    } else {
      using Impl = typename std::variant_alternative_t<start_idx, PirNodeImpl>;
      (*map)[typeid(Impl)] = &This::template ConvertAxprInstanceToPirNode<Impl>;
      InsertEntries<start_idx + 1>(map);
    }
  }

  template <typename T>
  static adt::Result<PirNode> ConvertAxprInstanceToPirNode(
      const axpr::BuiltinClassInstance<axpr::Value>& instance) {
    return PirNode{std::any_cast<T>(instance.instance)};
  }
};

}  // namespace ap::paddle
