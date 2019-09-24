// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_compatible_info.h"
#include <iostream>
#include <vector>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

/* first version >= second version return true */
inline bool CompareVersion(const int64_t first, const int64_t second) {
  return first >= second;
}

void OpCompatibleMap::InitOpCompatibleMap() {
  op_compatible_map_["sequence_pad"] = {1006000, OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["sequence_unpad"] = {1006000, OpCompatibleType::DEFIN_NOT};

  op_compatible_map_["reshape2"] = {1006000, OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["slice"] = {1006000, OpCompatibleType::possible};
  op_compatible_map_["expand"] = {1006000, OpCompatibleType::possible};

  op_compatible_map_["layer_norm"] = {1006000, OpCompatibleType::bug_fix};
}

CompatibleInfo OpCompatibleMap::GetOpCompatibleInfo(std::string op_name) const {
  auto it = op_compatible_map_.find(op_name);
  if (it != op_compatible_map_.end()) {
    return it->second;
  } else {
    return {default_required_version_, OpCompatibleType::DEFIN_NOT};
  }
}

OpCompatibleType OpCompatibleMap::IsRequireMiniVersion(
    std::string op_name, const int64_t current_version) const {
  auto it = op_compatible_map_.find(op_name);
  if (it != op_compatible_map_.end()) {
    if (CompareVersion(current_version, it->second.GetRequiredVersion())) {
      return OpCompatibleType::compatible;
    } else {
      return it->second.GetOpCompatibleType();
    }

  } else {
    if (CompareVersion(current_version, default_required_version_)) {
      return OpCompatibleType::compatible;
    } else {
      return OpCompatibleType::DEFIN_NOT;
    }
  }
}

}  // namespace framework
}  // namespace paddle
