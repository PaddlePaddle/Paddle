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

#include <vector>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/serializable_value.h"

namespace ap::registry {

struct AccessTopoDrrPassRegistryItemImpl {
  std::string access_topo_drr_pass_name;
  std::string pass_tag_name;
  int64_t nice;
  axpr::ClassAttrs<axpr::SerializableValue> cls;
};

ADT_DEFINE_RC(AccessTopoDrrPassRegistryItem, AccessTopoDrrPassRegistryItemImpl);

}  // namespace ap::registry
