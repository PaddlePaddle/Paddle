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

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/serialize_deserialize/include/schema.h"
#include "paddle/fluid/pir/serialize_deserialize/include/third_party.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace pir {

void GetPatchCompressOpName(std::string *op_name) {
  if (*op_name == "builtin.parameter") {
    *op_name = 'p';
    return;
  }
  std::pair<std::string, std::string> name = getContentSplitByDot(*op_name);
  *op_name = pir::DialectIdMap::Instance()->GetCompressDialectId(name.first) +
             "." + name.second;
  return;
}

// Json BuildAttrJsonPatch(const YAML::Node &action);

Json GetAttrTypeJson(const YAML::Node &action);

Json BuildAttrJsonPatch(const YAML::Node &action) {
  Json j_attr_type;
  if (!action["type"].IsDefined() || !action["default"].IsDefined()) {
    j_attr_type = nullptr;
  } else {
    j_attr_type = GetAttrTypeJson(action);
  }
  return j_attr_type;
}

Json GetAttrTypeJson(const YAML::Node &action) {
  Json json;
  std::string dialect = "0.";
  auto at_name = action["type"].as<std::string>();
  if (at_name == "pir::BoolAttribute") {
    VLOG(8) << "Get BoolAttribute name.";
    json[ID] = dialect + pir::BoolAttribute::name();
    json[DATA] = action["default"].as<bool>();
  } else if (at_name == "pir::FloatAttribute") {
    VLOG(8) << "Get FloatAttribute name.";
    json[ID] = dialect + pir::FloatAttribute::name();
    json[DATA] = action["default"].as<float>();
  } else if (at_name == "pir::DoubleAttribute") {
    VLOG(8) << "Get DoubleAttribute name.";
    json[ID] = dialect + pir::DoubleAttribute::name();
    json[DATA] = action["default"].as<double>();
  } else if (at_name == "pir::Int32Attribute") {
    VLOG(8) << "Get Int32Attribute name.";
    json[ID] = dialect + pir::Int32Attribute::name();
    json[DATA] = action["default"].as<int32_t>();
  } else if (at_name == "pir::Int64Attribute") {
    VLOG(8) << "Get Int64Attribute name.";
    json[ID] = dialect + pir::Int64Attribute::name();
    json[DATA] = action["default"].as<int64_t>();
  } else if (at_name == "pir::IndexAttribute") {
    VLOG(8) << "Get IndexAttribute name.";
    json[ID] = dialect + pir::IndexAttribute::name();
    json[DATA] = action["default"].as<int64_t>();
  } else if (at_name == "pir::ArrayAttribute") {
    VLOG(8) << "Get ArrayAttribute name.";
    json[ID] = dialect + pir::ArrayAttribute::name();
    json[DATA] = Json::array();
    for (size_t i = 0; i < action["default"].size(); ++i) {
      YAML::Node array_value = action["default"][i];
      json[DATA].push_back(BuildAttrJsonPatch(array_value));
    }
  } else if (at_name == "pir::TypeAttribute") {
    VLOG(8) << "Get TypeAttribute name.";
    json[ID] = dialect + pir::TypeAttribute::name();
    json[DATA] =
        action["default"].as<std::string>();  // TODO(czy): type attribute
  } else if (at_name == "pir::TensorNameAttribute") {
    VLOG(8) << "Get TensorNameAttribute name.";
    json[ID] = dialect + pir::TensorNameAttribute::name();
    json[DATA] = action["default"].as<std::string>();
  } else if (at_name == "pir::Complex64Attribute") {
    VLOG(8) << "Get Complex64Attribute name.";
    json[ID] = dialect + pir::Complex64Attribute::name();
    json[DATA] = action["default"].as<float>();
  } else if (at_name == "pir::Complex128Attribute") {
    VLOG(8) << "Get Complex128Attribute name.";
    json[ID] = dialect + pir::Complex128Attribute::name();
    json[DATA] = action["default"].as<double>();
  } else if (at_name == "pir::StrAttribute") {
    VLOG(8) << "Get StrAttribute name.";
    json[ID] = dialect + pir::StrAttribute::name();
    json[DATA] = action["default"].as<std::string>();
  } else {
    dialect = "1.";
    if (at_name == "paddle::dialect::IntArrayAttribute") {
      VLOG(8) << "Get IntArrayAttribute name.";
      json[ID] = dialect + paddle::dialect::IntArrayAttribute::name();
    } else if (at_name == "paddle::dialect::ScalarAttribute") {
      VLOG(8) << "Get ScalarAttribute name.";
      json[ID] = dialect + paddle::dialect::ScalarAttribute::name();

    } else if (at_name == "paddle::dialect::DataTypeAttribute") {
      VLOG(8) << "Get DataTypeAttribute name.";
      json[ID] = dialect + paddle::dialect::DataTypeAttribute::name();
    } else if (at_name == "paddle::dialect::PlaceAttribute") {
      VLOG(8) << "Get PlaceAttribute name.";
      json[ID] = dialect + paddle::dialect::PlaceAttribute::name();
    }
    PADDLE_ENFORCE(false,
                   phi::errors::InvalidArgument(
                       "Unknown Attr %s in the OpPatches.", at_name));
  }

  return json;
}
Json YamlPaser(const std::string &yaml_file) {
  YAML::Node root = YAML::LoadFile(yaml_file);
  Json json_patch;
  if (!root.IsDefined()) {
    std::cout << "Not defined" << std::endl;
  } else {
    std::cout << root << std::endl;
  }
  if (!root["op_patches"].IsSequence()) {
    std::cout << "Not a sequence" << std::endl;
  }
  json_patch["op_patches"] = Json::array();
  for (size_t i = 0; i < root["op_patches"].size(); i++) {
    // if (!item.IsMap()){
    //     std::cout<<"Not a map"<<std::endl;
    // }
    // parse op_name
    YAML::Node node = root["op_patches"][i];
    auto op_name = node["op_name"].as<std::string>();
    GetPatchCompressOpName(&op_name);
    Json j_patch;
    j_patch["op_name"] = op_name;
    j_patch["patch"] = Json::object();
    // parse actions
    auto actions = node["actions"];

    for (size_t j = 0; j < actions.size(); j++) {
      YAML::Node action = actions[j];
      if (!action.IsMap()) {
        std::cout << "Not a map" << std::endl;
      }
      std::string default_type;
      std::string action_name = action["action"].as<std::string>();
      if (action_name == "add_attr" || action_name == "modify_attr" ||
          action_name == "delete_attr") {
        std::string attr_name = action["object"].as<std::string>();
        Json j_attr;
        j_attr[NAME] = attr_name;
        j_attr[ATTR_TYPE] = BuildAttrJsonPatch(action);
        j_patch["patch"][ATTRS].push_back(j_attr);
      } else if (action_name == "add_output_attr" ||
                 action_name == "modify_output_attr" ||
                 action_name == "delete_output_attr") {
        std::string attr_name = action["object"].as<std::string>();
        Json j_attr;
        j_attr[NAME] = attr_name;
        j_attr[ATTR_TYPE] = BuildAttrJsonPatch(action);
        j_patch["patch"][OPRESULTS_ATTRS].push_back(j_attr);
      }
    }
    json_patch["op_patches"].push_back(j_patch);
  }
  std::cout << json_patch << std::endl;
  return json_patch;
}
}  // namespace pir
