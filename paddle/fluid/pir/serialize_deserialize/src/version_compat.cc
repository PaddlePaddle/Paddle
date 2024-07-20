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

#include "paddle/fluid/pir/serialize_deserialize/include/version_compat.h"
#include <fstream>
#include "paddle/fluid/pir/serialize_deserialize/include/patch_util.h"

namespace pir {

void PatchBuilder::BuildPatch(const std::string& path) {
  patch_json = YamlParser(path);
  for (auto patch_info : patch_json["op_patches"]) {
    op_patches_[patch_info["op_name"]] = patch_info["patch"];
  }
}

Json PatchBuilder::GetJsonOpPatch(const std::string& name) {
  return op_patches_[name];
}

void PatchBuilder::ApplyOpPatches(const std::string& op_name,
                                  Json* json,
                                  Json patch) {
  if (op_name == PARAMETEROP) {
    // attr_name ; type
    // is_distributed; array(bool)
    // is_parameter; array(bool)
    // need_clip; array(bool)
    // parameter_name; string
    // persistable; array(bool)
    // stop_gradient; array(bool)
    // trainable; array(bool)
    for (auto item : patch[ATTRS]) {
      std::unordered_map<std::string, int> attrs = {{"is_distributed", 0},
                                                    {"is_parameter", 1},
                                                    {"need_clip", 2},
                                                    {"parameter_name", 3}};
      std::string attr_name = item[NAME].get<std::string>();
      int index = attrs.count(attr_name) == 0 ? attrs.size() : attrs[attr_name];
      if (item[ATTR_TYPE][ID] == "0.a_str") {
        json->at(ATTRS)[index] = item[ATTR_TYPE][DATA];
      } else if (item[ATTR_TYPE][ID] == "0.a_int") {
        auto data = item[ATTR_TYPE][DATA];
        json->at(ATTRS)[index] = static_cast<int32_t>(data);
      }
    }
    for (auto item : patch[OPRESULTS_ATTRS]) {
      std::unordered_map<std::string, int> attrs = {
          {"persistable", 0}, {"stop_gradient", 1}, {"trainable", 2}};
      std::string attr_name = item[NAME].get<std::string>();
      int index = attrs.count(attr_name) == 0 ? attrs.size() : attrs[attr_name];
      if (item[ATTR_TYPE][ID] == "a_int") {
        auto data = item[ATTR_TYPE][DATA];
        json->at(ATTRS)[index] = static_cast<int32_t>(data);
      }
    }
    return;
  }
  // for (auto item : patch[ATTRS]) {
  //   std::string attr_name = item[NAME].get<std::string>();
  // }
  json->merge_patch(patch);
}

}  // namespace pir
