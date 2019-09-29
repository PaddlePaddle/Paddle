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
#include <utility>
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

CompatibleInfo OpCompatibleMap::GetOpCompatibleInfo(std::string op_name) const {
  auto it = op_compatible_map_.find(op_name);
  if (it != op_compatible_map_.end()) {
    return it->second;
  } else {
    return {default_required_version_, OpCompatibleType::DEFIN_NOT};
  }
}

OpCompatibleType OpCompatibleMap::IsRequireMiniVersion(
    std::string op_name, std::string str_current_version) const {
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

bool OpCompatibleMap::ConvertToProto(proto::OpCompatibleMap* desc) const {
  desc->Clear();
  desc->set_default_required_version(default_required_version_);
  for (auto pair : op_compatible_map_) {
    const CompatibleInfo& info = pair.second;
    auto* pair_desc = desc->add_pair();
    pair_desc->set_op_name(pair.first);
    auto* info_desc = pair_desc->mutable_compatible_info();
    info_desc->set_version(info.required_version_);
    info_desc->set_type(
        static_cast<proto::CompatibleInfo_Type>(info.compatible_type_));
  }
  return true;
}

bool OpCompatibleMap::ReadFromProto(const proto::OpCompatibleMap& desc) {
  std::string version = desc.default_required_version();
  if (version.empty()) {
    LOG(INFO) << "The default operator required version is missing."
                 " Please update the model version.";
    return false;
  }
  op_compatible_map_.clear();
  default_required_version_ = desc.default_required_version();
  for (int i = 0; i < desc.pair_size(); ++i) {
    const auto& pair_desc = desc.pair(i);
    auto info_desc = pair_desc.compatible_info();
    CompatibleInfo info(info_desc.version(),
                        static_cast<OpCompatibleType>(info_desc.type()));
    std::pair<std::string, CompatibleInfo> pair(pair_desc.op_name(), info);
    op_compatible_map_.insert(pair);
  }
  return true;
}

}  // namespace framework
}  // namespace paddle
