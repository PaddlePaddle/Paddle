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

#include <map>
#include <string>

#pragma once

namespace paddle {
namespace framework {
struct CompatibleInfo {
  CompatibleInfo(std::string required_version, int compatible_type)
      : required_version_(required_version),
        compatible_type_(compatible_type) {}
  CompatibleInfo() {}

  // op required version, previous version not support
  std::string required_version_;
  /* op support previous version
   *  0 for support previus version
   *  1 for definitely can't support previous version
   *  2 for possibel can't support previous version
   *  3 for bug fix  can't support previous version
   *  4 for precision change */
  int compatible_type_;
};

class OpCompatibleMap {
 public:
  OpCompatibleMap() : default_required_version_("1.5.0") {}
  void InitOpCompatibleMap();

  CompatibleInfo GetOpCompatibleInfo(std::string op_name);

  /* IsRequireMiniVersion
   *  return type
   *  0 for support previus version
   *  1 for definitely not support previous version
   *  2 for possibel not support previous version
   *  3 for bug fix, not support previous version
   *  4 for precision change */
  int IsRequireMiniVersion(std::string op_name, std::string current_version);

  void SerializeToStr(std::string& str) {} /* NOLINT */
  void UnSerialize(const std::string& str) {}

  const std::string& GetDefaultRequiredVersion() {
    return default_required_version_;
  }

 private:
  std::map<std::string, CompatibleInfo> op_compatible_map_;

  std::string default_required_version_;
};

}  // namespace framework
}  // namespace paddle
