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

inline std::vector<int> ConvertStr2Int(const std::string& str_text) {
  auto vec_text = string::split_string<std::string>(str_text, ".");
  PADDLE_ENFORCE((vec_text.size() == 2 || vec_text.size() == 3),
                 "Input[%s] is not a right version format [1.6 or 1.6.0]",
                 str_text);

  std::vector<int> vec_res;
  vec_res.reserve(3);
  for (auto& val : vec_text) {
    vec_res.emplace_back(atoi(val.c_str()));
  }

  if (vec_res.size() == 2) {
    vec_res.emplace_back(0);
  }

  return vec_res;
}

/* first version >= second version return true */

inline bool CompareVersion(const std::string& str_first,
                           const std::string& str_second) {
  auto vec_first_version = ConvertStr2Int(str_first);
  auto vec_second_version = ConvertStr2Int(str_second);

  // first version id
  PADDLE_ENFORCE_EQ(
      vec_first_version.size(), vec_second_version.size(),
      "version information size not equal, first is [%d] second is [%d]",
      vec_first_version.size(), vec_second_version.size());

  for (size_t i = 0; i < vec_first_version.size() - 1; ++i) {
    if (vec_first_version[i] != vec_second_version[i]) {
      return vec_first_version[i] > vec_second_version[i];
    }
  }
  return vec_first_version[2] >= vec_second_version[2];
}

void OpCompatibleMap::InitOpCompatibleMap() {
  op_compatible_map_["sequence_pad"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["sequence_unpad"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};

  op_compatible_map_["reshape2"] = {"1.6.0", OpCompatibleType::DEFIN_NOT};
  op_compatible_map_["slice"] = {"1.6.0", OpCompatibleType::possible};
  op_compatible_map_["expand"] = {"1.6.0", OpCompatibleType::possible};

  op_compatible_map_["layer_norm"] = {"1.6.0", OpCompatibleType::bug_fix};
}

CompatibleInfo OpCompatibleMap::GetOpCompatibleInfo(std::string op_name) {
  auto it = op_compatible_map_.find(op_name);
  if (it != op_compatible_map_.end()) {
    return it->second;
  } else {
    return {default_required_version_, OpCompatibleType::DEFIN_NOT};
  }
}

OpCompatibleType OpCompatibleMap::IsRequireMiniVersion(
    std::string op_name, std::string str_current_version) {
  auto it = op_compatible_map_.find(op_name);
  if (it != op_compatible_map_.end()) {
    if (CompareVersion(str_current_version, it->second.required_version_)) {
      return OpCompatibleType::compatible;
    } else {
      return it->second.compatible_type_;
    }

  } else {
    if (CompareVersion(str_current_version, default_required_version_)) {
      return OpCompatibleType::compatible;
    } else {
      return OpCompatibleType::DEFIN_NOT;
    }
  }
}

}  // namespace framework
}  // namespace paddle
