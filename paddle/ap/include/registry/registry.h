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

#include <map>
#include <unordered_map>
#include <vector>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/registry/abstract_drr_pass_registry_item.h"
#include "paddle/ap/include/registry/access_topo_drr_pass_registry_item.h"
#include "paddle/ap/include/registry/classic_drr_pass_registry_item.h"

namespace ap::registry {

template <typename T>
using Key2Nice2Items = std::map<std::string, std::map<int64_t, std::vector<T>>>;

struct RegistryImpl {
  Key2Nice2Items<AbstractDrrPassRegistryItem> abstract_drr_pass_registry_items;
  Key2Nice2Items<ClassicDrrPassRegistryItem> classic_drr_pass_registry_items;
  Key2Nice2Items<AccessTopoDrrPassRegistryItem>
      access_topo_drr_pass_registry_items;

  bool operator==(const RegistryImpl& other) const { return this == &other; }
};

ADT_DEFINE_RC(Registry, RegistryImpl);

}  // namespace ap::registry

namespace ap::axpr {

template <>
struct TypeImpl<registry::Registry> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "Registry"; }
};

}  // namespace ap::axpr
