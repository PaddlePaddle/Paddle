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

#include "paddle/fluid/pir/serialize_deserialize/include/schema.h"
#include <filesystem>
#include "paddle/phi/core/enforce.h"
namespace pir {

std::pair<std::string, std::string> GetContentSplitByDot(
    const std::string& str) {
  size_t pos = str.find('.');
  if (pos == std::string::npos) {
    return {str, str};
  }
  return {str.substr(0, pos), str.substr(pos + 1)};
}

std::vector<std::string> GetOpDistAttr() { return {"op_dist_attr", "op_role"}; }

void GetCompressOpName(std::string* op_name) {
  std::pair<std::string, std::string> name = GetContentSplitByDot(*op_name);
  *op_name = pir::DialectIdMap::Instance()->GetCompressDialectId(name.first) +
             "." + name.second;
  return;
}
#define DECOMPRESS_DIALECT_ID(name) \
  pir::DialectIdMap::Instance()->GetDecompressDialectId(name)
void GetDecompressOpName(std::string* op_name) {
  std::pair<std::string, std::string> name = GetContentSplitByDot(*op_name);
  *op_name = DECOMPRESS_DIALECT_ID(name.first) + "." + name.second;
  return;
}

DialectIdMap* DialectIdMap::Instance() {
  static DialectIdMap map;
  return &map;
}
DialectIdMap::DialectIdMap() {
  insert(pir::BuiltinDialect::name(), "0");
  insert(paddle::dialect::OperatorDialect::name(), "1");
  insert(pir::ControlFlowDialect::name(), "2");
  insert(paddle::dialect::CustomOpDialect::name(), "3");
  insert(paddle::dialect::DistDialect::name(), "4");
}
void DialectIdMap::insert(const std::string& key, const std::string& value) {
  CompressDialect[key] = value;
  DecompressDialect[value] = key;
}

std::string DialectIdMap::GetCompressDialectId(const std::string& name) {
  if (CompressDialect.find(name) != CompressDialect.end()) {
    return CompressDialect[name];
  } else {
    VLOG(0) << "can't find dialect " << name
            << "'s compress id, return original dialectname, it's better to "
               "insert compress id in DialectIdMap() func";
    return name;
  }
  return "";
}

std::string DialectIdMap::GetDecompressDialectId(const std::string& id) {
  if (DecompressDialect.find(id) != DecompressDialect.end()) {
    return DecompressDialect[id];
  } else {
    PADDLE_ENFORCE(
        false,
        common::errors::InvalidArgument(
            "Unknown id %s for decompress dialect, pleace check your file",
            id));
  }
  return "";
}

uint64_t GetPirVersion() {
  std::string current_path = std::filesystem::current_path().string();
  std::string paddle_root =
      current_path.substr(0, current_path.find("Paddle") + 7);
  VLOG(8) << "Paddle path: " << paddle_root;
  std::filesystem::path patch_path =
      std::filesystem::path(paddle_root.c_str()) / "paddle" / "fluid" / "pir" /
      "serialize_deserialize" / "patch";
  VLOG(8) << "Patch path: " << patch_path;
  int version = 0;
  for (auto& v : std::filesystem::directory_iterator(patch_path)) {
    std::string filename = v.path().filename().string();
    std::string extension_name = v.path().extension().string();
    // 0.yaml for develop version
    if (filename == "0.yaml") {
      VLOG(8) << "Develop version: " << version;
      return 0;
    } else if (extension_name == ".yaml") {
      version = stoi(filename) > version ? stoi(filename) : version;
    }
  }
  VLOG(8) << "PIR version: " << version;
  return version;
}
uint64_t GetMaxReleasePirVersion() {
  std::string current_path = std::filesystem::current_path().string();
  std::string paddle_root =
      current_path.substr(0, current_path.find("Paddle") + 7);
  VLOG(8) << "Paddle path: " << paddle_root;
  std::filesystem::path patch_path =
      std::filesystem::path(paddle_root.c_str()) / "paddle" / "fluid" / "pir" /
      "serialize_deserialize" / "patch";
  VLOG(8) << "Patch path: " << patch_path;
  int version = 0;
  for (auto& v : std::filesystem::directory_iterator(patch_path)) {
    std::string filename = v.path().filename().string();
    std::string extension_name = v.path().extension().string();
    VLOG(8) << filename;
    if (extension_name == ".yaml") {
      version = stoi(filename) > version ? stoi(filename) : version;
    }
  }
  VLOG(8) << "Max Release PIR version: " << version;
  return version;
}

}  // namespace pir
