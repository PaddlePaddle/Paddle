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

#include "paddle/fluid/pir/serialize_deserialize/include/patch_util.h"
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/serialize_deserialize/include/schema.h"
#include "paddle/fluid/pir/serialize_deserialize/patch/patch.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

namespace pir {

Json BuildAttrJsonPatch(const YAML::Node &action) {
  Json j_attr_type;
  if (!action["type"].IsDefined() && !action["default"].IsDefined()) {
    j_attr_type = nullptr;
  } else {
    j_attr_type = GetAttrJson(action);
    j_attr_type = GetAttrJson(action);
  }
  return j_attr_type;
}
std::string GetAttrName(const YAML::Node &action) {
  Json j_attr;
  j_attr = GetAttrJson(action);
  return j_attr[ID];
}
Json GetAttrJson(const YAML::Node &action) {
  Json json;
  std::string dialect = DialectIdMap::Instance()->GetCompressDialectId(
                            pir::BuiltinDialect::name()) +
                        ".";
  std::string at_name;
  if (action.IsScalar()) {
    at_name = action.as<std::string>();
    VLOG(8) << "Get old Type name." << at_name;
  } else {
    at_name = action["type"].as<std::string>();
    VLOG(8) << "Get new Type name." << at_name;
  }
  if (at_name == "pir::BoolAttribute") {
    VLOG(8) << "Get BoolAttribute name.";
    json[ID] = dialect + pir::BoolAttribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<bool>();
    }
  } else if (at_name == "pir::FloatAttribute") {
    VLOG(8) << "Get FloatAttribute name.";
    json[ID] = dialect + pir::FloatAttribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<float>();
    }
  } else if (at_name == "pir::DoubleAttribute") {
    VLOG(8) << "Get DoubleAttribute name.";
    json[ID] = dialect + pir::DoubleAttribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<double>();
    }
  } else if (at_name == "pir::Int32Attribute") {
    VLOG(8) << "Get Int32Attribute name.";
    json[ID] = dialect + pir::Int32Attribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<int32_t>();
    }
  } else if (at_name == "pir::Int64Attribute") {
    VLOG(8) << "Get Int64Attribute name.";
    json[ID] = dialect + pir::Int64Attribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<int64_t>();
    }
  } else if (at_name == "pir::IndexAttribute") {
    VLOG(8) << "Get IndexAttribute name.";
    json[ID] = dialect + pir::IndexAttribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<int64_t>();
    }
  } else if (at_name == "pir::ArrayAttribute") {
    VLOG(8) << "Get ArrayAttribute name.";
    json[ID] = dialect + pir::ArrayAttribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = Json::array();
      for (size_t i = 0; i < action["default"].size(); ++i) {
        YAML::Node array_value = action["default"][i];
        json[DATA].push_back(BuildAttrJsonPatch(array_value));
      }
    }
  } else if (at_name == "pir::TypeAttribute") {
    VLOG(8) << "Get TypeAttribute name.";
    json[ID] = dialect + pir::TypeAttribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<std::string>();
    }  // TODO(czy): type attribute
  } else if (at_name == "pir::TensorNameAttribute") {
    VLOG(8) << "Get TensorNameAttribute name.";
    json[ID] = dialect + pir::TensorNameAttribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<std::string>();
    }
  } else if (at_name == "pir::Complex64Attribute") {
    VLOG(8) << "Get Complex64Attribute name.";
    json[ID] = dialect + pir::Complex64Attribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<float>();
    }
  } else if (at_name == "pir::Complex128Attribute") {
    VLOG(8) << "Get Complex128Attribute name.";
    json[ID] = dialect + pir::Complex128Attribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<double>();
    }
  } else if (at_name == "pir::StrAttribute") {
    VLOG(8) << "Get StrAttribute name.";
    json[ID] = dialect + pir::StrAttribute::name();
    if (!action.IsScalar() && action["default"].IsDefined()) {
      json[DATA] = action["default"].as<std::string>();
    }
  } else {  // TODO(czy): add data patch for other attributes
    dialect = DialectIdMap::Instance()->GetCompressDialectId(
                  paddle::dialect::OperatorDialect::name()) +
              ".";
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
                   common::errors::InvalidArgument(
                       "Unknown Attr %s in the OpPatches.", at_name));
  }
  return json;
}

Json BuildTypeJsonPatch(const YAML::Node &action) {
  Json json = GetTypeJson(action);
  return json;
}

std::string GetTypeName(const YAML::Node &action) {
  Json json = GetTypeJson(action);
  return json[ID];
}

Json GetTypeJson(const YAML::Node &action) {
  Json json;
  std::string builtin_dialect = DialectIdMap::Instance()->GetCompressDialectId(
                                    pir::BuiltinDialect::name()) +
                                ".";
  std::string op_dialect = DialectIdMap::Instance()->GetCompressDialectId(
                               paddle::dialect::OperatorDialect::name()) +
                           ".";
  std::string cf_dialect = DialectIdMap::Instance()->GetCompressDialectId(
                               pir::ControlFlowDialect::name()) +
                           ".";
  std::string type_name = "";
  if (action.IsScalar()) {
    type_name = action.as<std::string>();
    VLOG(8) << "Get old Type name." << type_name;
  } else {
    type_name = action["type"].as<std::string>();
    VLOG(8) << "Get new Type name." << type_name;
  }
  if (type_name == "pir::BoolType") {
    VLOG(8) << "Get BoolType name.";
    json[ID] = builtin_dialect + pir::BoolType::name();
  } else if (type_name == "pir::BFloat16Type") {
    VLOG(8) << "Get BFloat16Type name.";
    json[ID] = builtin_dialect + pir::BFloat16Type::name();
  } else if (type_name == "pir::Float16Type") {
    VLOG(8) << "Get Float16Type name.";
    json[ID] = builtin_dialect + pir::Float16Type::name();
  } else if (type_name == "pir::Float32Type") {
    VLOG(8) << "Get Float32Type name.";
    json[ID] = builtin_dialect + pir::Float32Type::name();
  } else if (type_name == "pir::Float64Type") {
    VLOG(8) << "Get Float64Type name.";
    json[ID] = builtin_dialect + pir::Float64Type::name();
  } else if (type_name == "pir::Int8Type") {
    VLOG(8) << "Get Int8Type name.";
    json[ID] = builtin_dialect + pir::Int8Type::name();
  } else if (type_name == "pir::UInt8Type") {
    VLOG(8) << "Get UInt8Type name.";
    json[ID] = builtin_dialect + pir::UInt8Type::name();
  } else if (type_name == "pir::Int16Type") {
    VLOG(8) << "Get Int16Type name.";
    json[ID] = builtin_dialect + pir::Int16Type::name();
  } else if (type_name == "pir::Int32Type") {
    VLOG(8) << "Get Int32Type name.";
    json[ID] = builtin_dialect + pir::Int32Type::name();
  } else if (type_name == "pir::Int64Type") {
    VLOG(8) << "Get Int64Type name.";
    json[ID] = builtin_dialect + pir::Int64Type::name();
  } else if (type_name == "pir::IndexType") {
    VLOG(8) << "Get IndexType name.";
    json[ID] = builtin_dialect + pir::IndexType::name();
  } else if (type_name == "pir::Complex64Type") {
    VLOG(8) << "Get Complex64Type name.";
    json[ID] = builtin_dialect + pir::Complex64Type::name();
  } else if (type_name == "pir::Complex128Type") {
    VLOG(8) << "Get Complex128Type name.";
    json[ID] = builtin_dialect + pir::Complex128Type::name();
  } else if (type_name == "pir::VectorType") {
    VLOG(8) << "Get VectorType name.";
    json[ID] = builtin_dialect + pir::VectorType::name();
    json[DATA] = Json::array();
    for (size_t i = 0; i < action["default"].size(); i++) {
      YAML::Node array_value = action["default"][i];
      json[DATA].push_back(BuildTypeJsonPatch(array_value));
    }
  } else if (type_name == "pir::DenseTensorType") {
    VLOG(8) << "Get DenseTensorType name.";
    json[ID] = builtin_dialect + pir::DenseTensorType::name();
    Json content = Json::array();
    YAML::Node tensor_value = action["default"];
    content.push_back(BuildTypeJsonPatch(tensor_value[0]));

    content.push_back(tensor_value[1].as<std::vector<int>>());  // Dims

    content.push_back(tensor_value[2].as<std::string>());  // DataLayout

    content.push_back(
        tensor_value[3].as<std::vector<std::vector<int>>>());  // LoD

    content.push_back(tensor_value[4].as<int>());  // offset
    json[DATA] = content;
  } else if (type_name == "pir::StackType") {
    VLOG(8) << "Get StackType name.";
    json[ID] = cf_dialect + pir::StackType::name();
  } else if (type_name == "pir::InletType") {
    VLOG(8) << "Get InletType name.";
    json[ID] = cf_dialect + pir::InletType::name();
  } else if (type_name == "pir::OutletType") {
    VLOG(8) << "Get OutletType name.";
    json[ID] = cf_dialect + pir::OutletType::name();
  }
  return json;
}

Json ParseOpPairPatches(const YAML::Node &root) {
  Json json_patch = Json::array();
  for (size_t i = 0; i < root.size(); i++) {
    // parse op_name
    YAML::Node node = root[i];
    Json j_patch;
    j_patch["op_pair"] = Json::array();
    for (auto name : node["op_pair"]) {
      auto op_name = name.as<std::string>();
      if (op_name == pir::ParameterOp::name()) {
        op_name = PARAMETEROP;
      } else {
        GetCompressOpName(&op_name);
      }
      VLOG(8) << "Op_pair_name: " << name;
      j_patch["op_pair"].push_back(op_name);
    }
    j_patch[PATCH] = Json::object();
    j_patch[PATCH]["op_pair"] = j_patch["op_pair"];
    // parse actions
    auto actions = node["actions"];
    for (size_t j = 0; j < actions.size(); j++) {
      YAML::Node action = actions[j];
      std::string default_type;
      std::string action_name = action["action"].as<std::string>();
      VLOG(8) << "Patch action_name: " << action_name;
      if (action_name == "add_value") {
        VLOG(8) << "Patch for adding values.";
        int out_id = action["object"][0].as<int>();
        int in_id = action["object"][1].as<int>();
        Json j_add_out;
        j_add_out[ID] = out_id;
        j_add_out[TYPE_TYPE] = BuildTypeJsonPatch(action);
        j_patch[PATCH][OPRESULTS][ADD].push_back(j_add_out);
        Json j_add_in;
        j_add_in[ID] = in_id;
        j_patch[PATCH][OPOPERANDS][ADD].push_back(j_add_in);
      } else if (action_name == "delete_value") {
        VLOG(8) << "Patch for deleting values.";
        int out_id = action["object"][0].as<int>();
        int in_id = action["object"][1].as<int>();
        Json j_del_out;
        j_del_out[ID] = out_id;
        j_patch[PATCH][OPRESULTS][DELETE].push_back(j_del_out);
        Json j_del_in;
        j_del_in[ID] = in_id;
        j_patch[PATCH][OPOPERANDS][DELETE].push_back(j_del_in);
      }
    }
    json_patch.push_back(j_patch);
  }
  VLOG(8) << "Op pair patches: " << json_patch;
  return json_patch;
}

Json ParseOpPatches(const YAML::Node &root) {
  Json json_patch = Json::array();
  for (size_t i = 0; i < root.size(); i++) {
    // parse op_name
    YAML::Node node = root[i];
    auto op_name = node["op_name"].as<std::string>();
    if (op_name == pir::ParameterOp::name()) {
      op_name = PARAMETEROP;
    } else {
      GetCompressOpName(&op_name);
    }
    VLOG(8) << "Parse patches for " << op_name;
    Json j_patch;
    j_patch["op_name"] = op_name;
    j_patch[PATCH] = Json::object();
    // parse actions
    auto actions = node["actions"];

    for (size_t j = 0; j < actions.size(); j++) {
      YAML::Node action = actions[j];
      if (!action.IsMap()) {
        VLOG(8) << "Not a map";
      }
      std::string default_type;
      std::string action_name = action["action"].as<std::string>();
      VLOG(8) << "Patch action_name: " << action_name;
      if (action_name == "add_attr" || action_name == "modify_attr" ||
          action_name == "delete_attr") {
        VLOG(8) << "Patch for attrs.";
        std::string attr_name = action["object"].as<std::string>();
        Json j_attr;
        j_attr[NAME] = attr_name;
        j_attr[ATTR_TYPE] = BuildAttrJsonPatch(action);
        if (action_name == "add_attr") {
          Json j_add = Json::object();
          j_add[ADD] = j_attr;
          j_patch[PATCH][ATTRS].push_back(j_add);
        } else {
          j_patch[PATCH][ATTRS].push_back(j_attr);
        }
      } else if (action_name == "add_output_attr" ||
                 action_name == "modify_output_attr" ||
                 action_name == "delete_output_attr") {
        VLOG(8) << "Patch for output_attrs.";
        std::string attr_name = action["object"].as<std::string>();
        Json j_attr;
        j_attr[NAME] = attr_name;
        j_attr[ATTR_TYPE] = BuildAttrJsonPatch(action);
        if (action_name == "add_output_attr") {
          Json j_add = Json::object();
          j_add[ADD] = j_attr;
          j_patch[PATCH][OPRESULTS_ATTRS].push_back(j_add);
        } else {
          j_patch[PATCH][OPRESULTS_ATTRS].push_back(j_attr);
        }
      } else if (action_name == "modify_attr_name" ||
                 action_name == "modify_output_attr_name") {
        VLOG(8) << "Patch for modify attr name.";
        std::string old_name = action["object"].as<std::string>();
        std::string new_name = action["default"].as<std::string>();
        Json j_attr;
        j_attr[NAME] = old_name;
        j_attr[NEW_NAME] = new_name;
        std::string col =
            action_name == "modify_attr_name" ? ATTRS : OPRESULTS_ATTRS;
        j_patch[PATCH][col].push_back(j_attr);
      } else if (action_name == "delete_input") {
        VLOG(8) << "Patch for delete_input";
        Json j_input;
        int op_id = action["object"].as<int>();
        j_input[ID] = op_id;
        j_patch[PATCH][OPOPERANDS][DELETE].push_back(j_input);
      } else if (action_name == "add_output") {
        VLOG(8) << "Patch for add_output";
        Json j_output;
        int op_id = action["object"].as<int>();
        j_output[ID] = op_id;
        j_output[TYPE_TYPE] = BuildTypeJsonPatch(action);
        j_patch[PATCH][OPRESULTS][ADD].push_back(j_output);
      } else if (action_name == "modify_output_type") {
        VLOG(8) << "Patch for modify_output_type";
        int op_id = action["object"].as<int>();
        Json j_type;
        j_type[ID] = op_id;
        j_type[TYPE_TYPE] = BuildTypeJsonPatch(action);
        j_patch[PATCH][OPRESULTS][UPDATE].push_back(j_type);
      } else if (action_name == "modify_name") {
        VLOG(8) << "Patch for modify_name";
        std::string op_name = action["default"].as<std::string>();
        GetCompressOpName(&op_name);
        j_patch[PATCH][NEW_NAME] = op_name;
      }
    }
    json_patch.push_back(j_patch);
  }
  VLOG(8) << "Op patches built successfully: " << json_patch;
  return json_patch;
}

Json ParseTypePatches(const YAML::Node &root) {
  Json json_patch = Json::array();
  for (size_t i = 0; i < root.size(); i++) {
    // parse op_name
    YAML::Node node = root[i];
    YAML::Node name = node["type_name"];
    VLOG(8) << "Build patch for: " << name;
    auto type_name = GetTypeName(name);
    VLOG(8) << "Type name after compressing: " << type_name;
    Json j_patch;
    j_patch["type_name"] = type_name;
    j_patch[PATCH] = Json::object();
    auto actions = node["actions"];
    for (size_t j = 0; j < actions.size(); j++) {
      YAML::Node action = actions[j];
      std::string action_name = action["action"].as<std::string>();
      if (action_name == "modify_name") {
        j_patch[PATCH][NEW_NAME] = GetTypeName(action);
      } else if (action_name == "delete_type") {
        j_patch[PATCH][NEW_NAME] = "";
      }
    }
    json_patch.push_back(j_patch);
    VLOG(8) << type_name << " type patch: " << json_patch;
  }

  return json_patch;
}

Json ParseAttrPatches(const YAML::Node &root) {
  Json json_patch = Json::array();
  for (size_t i = 0; i < root.size(); i++) {
    // parse op_name
    YAML::Node node = root[i];
    YAML::Node name = node["attr_name"];
    VLOG(8) << "node: " << node << "attr_name: " << name;
    auto attr_name = GetAttrName(name);
    VLOG(8) << attr_name;
    Json j_patch;
    j_patch["attr_name"] = attr_name;
    j_patch[PATCH] = Json::object();
    auto actions = node["actions"];
    for (size_t j = 0; j < actions.size(); j++) {
      YAML::Node action = actions[j];
      std::string action_name = action["action"].as<std::string>();
      if (action_name == "modify_name") {
        j_patch[PATCH][NEW_NAME] = GetAttrName(action);
      } else if (action_name == "delete_attr") {
        j_patch[PATCH][NEW_NAME] = "";
      }
    }
    json_patch.push_back(j_patch);
  }
  return json_patch;
}

Json YamlParser(const std::string &version, const std::string &yaml_file) {
  YAML::Node root;
  std::ifstream fin;
  if (yaml_file.empty()) {
    root = YAML::Load(yaml_files.at(version));
  } else {
    VLOG(8) << yaml_file;
    fin.open(yaml_file);
    if (!fin) {
      VLOG(8) << yaml_file << " is not fin and return empty. ";
      return Json::object();
    }
    root = YAML::Load(fin);
  }
  Json json_patch;
  if (!root.IsDefined()) {
    VLOG(8) << "Not defined";
  } else {
    VLOG(8) << root;
  }
  if (!root["op_patches"].IsSequence()) {
    VLOG(8) << "Not a sequence";
  }
  Yaml op_pair_patch = root["op_pair_patches"];
  json_patch["op_pair_patches"] = ParseOpPairPatches(op_pair_patch);
  Yaml op_patch = root["op_patches"];
  json_patch["op_patches"] = ParseOpPatches(op_patch);
  VLOG(8) << "Finish op json_patch: " << json_patch;
  Yaml type_patch = root["type_patches"];
  json_patch["type_patches"] = ParseTypePatches(type_patch);
  VLOG(8) << "Finish type json_patch: " << json_patch;
  Yaml attr_patch = root["attr_patches"];
  json_patch["attr_patches"] = ParseAttrPatches(attr_patch);
  VLOG(8) << "Finish attr json_patch: " << json_patch;
  if (fin) {
    fin.close();
  }
  return json_patch;
}
}  // namespace pir
