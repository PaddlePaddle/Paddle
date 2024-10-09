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
#include "paddle/fluid/pir/serialize_deserialize/include/patch_util.h"
namespace pir {

void PatchBuilder::BuildPatch(uint64_t pir_version,
                              uint64_t max_version,
                              const std::string& path) {
  VLOG(6) << "Begin building patches... ";
  for (auto v = file_version_; v <= pir_version; v++) {
    std::string file_path = "";
    std::string file_name = std::to_string(v % max_version);
    if (!path.empty()) {
      std::filesystem::path p(path.c_str());
      std::filesystem::path patch_path = p / file_name;
      patch_path += ".yaml";
      VLOG(8) << "Patch file: " << patch_path;
      file_path = patch_path.string();
    }
    patch_json = YamlParser(file_name, file_path);
    VLOG(8) << "Build version " << v << " patch: " << patch_json;
    for (auto patch_info : patch_json["op_pair_patches"]) {
      op_pair_patches_.insert(patch_info[PATCH]);
      VLOG(8) << "merge op_pair patch: " << op_pair_patches_;
    }
    for (auto patch_info : patch_json["op_patches"]) {
      VLOG(8) << "merge op patch: " << patch_info["op_name"];
      if (op_patches_.count(patch_info["op_name"])) {
        Json op_patch_orig = op_patches_[patch_info["op_name"]];
        Json op_patch_new = patch_info[PATCH];
        for (auto item : op_patch_new.items()) {
          std::string key = item.key();
          Json value = item.value();
          if (op_patch_orig.count(key) == 0) {
            op_patch_orig[key] = value;
          } else {
            Json value_orig = op_patch_orig[key];
            if (key == OPOPERANDS || key == OPRESULTS) {
              for (auto action : value.items()) {
                std::string action_key = action.key();
                Json action_value = action.value();
                if (value_orig.count(action_key) == 0) {
                  value_orig[action_key] = action_value;
                } else {
                  value_orig[action_key].insert(value_orig[action_key].end(),
                                                action_value.begin(),
                                                action_value.end());
                }
              }
            } else if (key == NEW_NAME) {
              value_orig = value;
            } else {
              value_orig.insert(value_orig.end(), value.begin(), value.end());
            }
          }
        }
        VLOG(8) << "merge op patch: " << op_patches_[patch_info["op_name"]];
      } else {
        op_patches_[patch_info["op_name"]] = patch_info[PATCH];
      }
    }
    for (auto patch_info : patch_json["type_patches"]) {
      type_patches_[patch_info["type_name"]].update(patch_info[PATCH], true);
      VLOG(8) << "merge type patch: " << type_patches_;
    }
    for (auto patch_info : patch_json["attr_patches"]) {
      attr_patches_[patch_info["attr_name"]].update(patch_info[PATCH], true);
      VLOG(8) << "merge attr patch: " << attr_patches_;
    }
  }
  VLOG(8) << "Finish build op_pair_patches_: " << op_pair_patches_;
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
      if (item.count(ADD)) {
        op_attr_patch[ADD_ATTRS].push_back(item.at(ADD));
      } else {
        op_attr_patch[item[NAME]].push_back(item);
      }
    }
  }
  if (op_patch.count(OPRESULTS_ATTRS)) {
    for (Json item : op_patch[OPRESULTS_ATTRS]) {
      if (item.count(ADD)) {
        op_attr_patch[ADD_OPRESULTS_ATTRS].push_back(item.at(ADD));
      } else {
        op_attr_patch[item[NAME]].push_back(item);
      }
    }
  }
  return op_attr_patch;
}

void PatchBuilder::ApplyOpPairPatches(int64_t* id) {
  int max_id = *id;
  VLOG(6) << "Start apply op_pair patches...";
  VLOG(8) << "Max id before applying op pair patches: " << max_id;
  for (auto item : op_pair_patches_) {
    std::string op1 = item["op_pair"][0];
    std::string op2 = item["op_pair"][1];
    Json op1_patch = item[OPRESULTS];
    Json op2_patch = item[OPOPERANDS];
    for (uint64_t i = 0; i < op1_patch[ADD].size(); ++i) {
      max_id++;
      op1_patch[ADD][i][VALUE_ID] = max_id;
      op2_patch[ADD][i][VALUE_ID] = max_id;
      op_patches_[op1][OPRESULTS][ADD].push_back(op1_patch[ADD][i]);
      op_patches_[op2][OPOPERANDS][ADD].push_back(op2_patch[ADD][i]);
    }
    for (uint64_t i = 0; i < op1_patch[DELETE].size(); ++i) {
      op_patches_[op1][OPRESULTS][DELETE].push_back(op1_patch[DELETE][i]);
      op_patches_[op2][OPOPERANDS][DELETE].push_back(op2_patch[DELETE][i]);
    }
  }
  *id = max_id;
  VLOG(8) << "Op patches after applying op pair patches: \n" << op_patches_;
  VLOG(8) << "Max id after applying op pair patches: " << max_id;
  VLOG(6) << "Finish apply op_pair patches... ";
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
      if (item[ATTR_TYPE][ID] == DialectIdMap::Instance()->GetCompressDialectId(
                                     pir::BuiltinDialect::name()) +
                                     "." + pir::StrAttribute::name()) {
        json->at(ATTRS)[index] = item[ATTR_TYPE][DATA];
      } else if (item[ATTR_TYPE][ID] ==
                 DialectIdMap::Instance()->GetCompressDialectId(
                     pir::BuiltinDialect::name()) +
                     "." + pir::BoolAttribute::name()) {
        auto data = item[ATTR_TYPE][DATA];
        json->at(ATTRS)[index] = static_cast<int32_t>(data);
      }
    }
    for (auto item : patch[OPRESULTS_ATTRS]) {
      std::unordered_map<std::string, int> attrs = {
          {"persistable", 0}, {"stop_gradient", 1}, {"trainable", 2}};
      std::string attr_name = item[NAME].get<std::string>();
      int index = attrs.count(attr_name) == 0 ? attrs.size() : attrs[attr_name];
      if (item[ATTR_TYPE][ID] == DialectIdMap::Instance()->GetCompressDialectId(
                                     pir::BuiltinDialect::name()) +
                                     "." + pir::BoolAttribute::name()) {
        auto data = item[ATTR_TYPE][DATA];
        json->at(OPRESULTS_ATTRS)[index] = static_cast<int32_t>(data);
      }
    }
    return;
  }
  // Deal with io patches
  if (patch.contains(OPOPERANDS)) {
    Json* json_in = &json->at(OPOPERANDS);
    Json in_patch = patch[OPOPERANDS];
    for (auto item : in_patch[ADD]) {
      int id = item[ID].get<int>();
      auto index = json_in->begin() + id;
      VLOG(8) << "Add index: " << id;
      item.erase(ID);
      json_in->insert(index, item);
      VLOG(8) << "ADD output: " << json_in;
    }
    for (auto item : in_patch[DELETE]) {
      int id = item[ID].get<int>();
      json_in->erase(id);
    }
  }
  if (patch.contains(OPRESULTS)) {
    Json* json_out = &json->at(OPRESULTS);
    Json out_patch = patch[OPRESULTS];
    VLOG(8) << "out patch: " << out_patch;
    for (auto item : out_patch[UPDATE]) {
      int id = item[ID].get<int>();
      item.erase(ID);
      json_out->at(id)[TYPE_TYPE] = item[TYPE_TYPE];
    }
    for (auto item : out_patch[ADD]) {
      int id = item[ID].get<int>();
      auto index = json_out->begin() + id;
      VLOG(8) << "Add index: " << id;
      item.erase(ID);
      json_out->insert(index, item);
      VLOG(8) << "ADD output: " << json_out;
    }
    for (auto item : out_patch[DELETE]) {
      int id = item[ID].get<int>();
      json_out->erase(id);
    }
  }
}

void PatchBuilder::ApplyTypePatches(const std::string& type_name,
                                    Json* json,
                                    Json patch) {
  json->at(ID) = patch[NEW_NAME];
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
  std::string name = attr_name;
  for (auto item : patch) {
    if (item.contains(NEW_NAME)) {
      name = item[NEW_NAME].get<std::string>();
    } else {
      json->merge_patch(item);
    }
  }
  json->at(NAME) = name;
}

void PatchBuilder::ApplyAttrTypePatches(const std::string& attr_name,
                                        Json* json,
                                        Json patch) {
  if (patch.contains(NEW_NAME)) {
    json->at(ID) = patch[NEW_NAME];
  }
}
}  // namespace pir
