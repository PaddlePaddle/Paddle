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
#include "paddle/phi/core/enforce.h"
namespace pir {

std::pair<std::string, std::string> getContentSplitByDot(
    const std::string& str) {
  size_t pos = str.find('.');
  if (pos == std::string::npos) {
    return {str, str};
  }
  return {str.substr(0, pos), str.substr(pos + 1)};
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
        phi::errors::InvalidArgument(
            "Unknown id %s for decompress dialect, pleace check your file",
            id));
  }
  return "";
}

}  // namespace pir
