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
#include <filesystem>
#include <fstream>
#include "paddle/fluid/pir/serialize_deserialize/include/patch_util.h"
namespace pir {

void PatchBuilder::BuildPatch(const std::string& path) {
  VLOG(6) << "Begin building patches... ";
  // Set max_version to the max version number of release pir plus 1.
  auto max_version = GetMaxReleasePirVersion() + 1;
  // If pir_version_ is not 0, we will build patch from file_version_ to
  // pir_version_; If pir_version_ is 0, we will first build patch from
  // file_version_ to max_version, and then add 0.yaml to the end.
  auto pir_version = pir_version_ == 0 ? max_version : pir_version_;
  VLOG(6) << "file_version_: " << file_version_
          << ", pir_version: " << pir_version;
  for (auto v = file_version_; v <= pir_version; v++) {
    std::filesystem::path p(path.c_str());
    std::filesystem::path patch_path = p / std::to_string(v % max_version);
    patch_path += ".yaml";
    VLOG(8) << "Patch file: " << patch_path;
    patch_json = YamlParser(patch_path.string());
    VLOG(8) << "Build version " << v << " patch: " << patch_json;
    for (auto patch_info : patch_json["op_patches"]) {
      if (op_patches_.count(patch_info["op_name"])) {
        Json op_patch_orig = op_patches_[patch_info["op_name"]];
        Json op_patch_new = patch_info["patch"];
        for (auto item : op_patch_new.items()) {
          std::string key = item.key();
          Json value = item.value();
          if (op_patch_orig.count(key) == 0) {
            op_patch_orig[key] = value;
          } else {
            Json value_orig = op_patch_orig[key];
            value_orig.insert(value_orig.end(), value.begin(), value.end());
          }
        }
      } else {
        op_patches_[patch_info["op_name"]] = patch_info["patch"];
      }
    }
    for (auto patch_info : patch_json["type_patches"]) {
      type_patches_[patch_info["type_name"]].update(patch_info["patch"], true);
    }
    for (auto patch_info : patch_json["attr_patches"]) {
      attr_patches_[patch_info["attr_name"]].update(patch_info["patch"], true);
    }
  }
  VLOG(8) << "Finish build op_patches_: " << op_patches_;
  VLOG(8) << "Finish build type_patches_: " << type_patches_;
  VLOG(8) << "Finish build attr_patches_: " << attr_patches_;
  VLOG(6) << "Finish building patches... ";
}

std::unordered_map<std::string, Json> PatchBuilder::GetOpAttrPatchMap(
    const Json op_patch) {
  std::unordered_map<std::string, Json> op_attr_patch;
  if (op_patch.count(ATTRS)) {
    for (Json item : op_patch[ATTRS]) {
      op_attr_patch[item[NAME]] = item;
    }
  }
  if (op_patch.count(OPRESULTS_ATTRS)) {
    for (Json item : op_patch[OPRESULTS_ATTRS]) {
      op_attr_patch[item[NAME]] = item;
    }
  }
  return op_attr_patch;
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
  // Deal with io patches
  if (patch.contains(OPOPERANDS)) {
    Json* json_in = &json->at(OPOPERANDS);
    Json in_patch = patch[OPOPERANDS];
    for (auto item : in_patch["DELETE"]) {
      int id = item.get<int>();
      json_in->erase(id);
    }
  }
  if (patch.contains(OPRESULTS)) {
    Json* json_out = &json->at(OPRESULTS);
    Json out_patch = patch[OPRESULTS];
    VLOG(8) << "out patch: " << out_patch;
    for (auto item : out_patch["ADD"]) {
      int id = item[VALUE_ID].get<int>();
      auto index = json_out->begin() + id;
      VLOG(8) << "Add index: " << id;
      item.erase(VALUE_ID);
      json_out->insert(index, item);
      VLOG(8) << "ADD output: " << json_out;
    }
  }
}

void PatchBuilder::ApplyTypePatches(const std::string& type_name,
                                    Json* json,
                                    Json patch) {
  json->at(ID) = patch["NEW_NAME"];
  if (type_name == pir::DenseTensorType::name()) {
    std::string name = json->at(DATA).at(0).get<std::string>();
    if (HasTypePatch(name)) {
      Json patch_json = type_patches_[name];
      ApplyTypePatches(name, &json->at(DATA).at(0), patch_json);
    }
  }
}

void PatchBuilder::ApplyAttrPatches(const std::string& attr_name,
                                    Json* json,
                                    Json patch) {
  if (patch.contains("NEW_NAME")) {
    json->at(NAME) = patch["NEW_NAME"];
  } else {
    json->merge_patch(patch);
  }
  VLOG(8) << attr_name << ": " << json;
}

void PatchBuilder::ApplyAttrTypePatches(const std::string& attr_name,
                                        Json* json,
                                        Json patch) {
  if (patch.contains("NEW_NAME")) {
    json->at(ID) = patch["NEW_NAME"];
  }
}
}  // namespace pir
