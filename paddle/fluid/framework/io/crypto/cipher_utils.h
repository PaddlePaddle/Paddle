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

#pragma once

#include <sstream>
#include <string>
#include <unordered_map>

namespace paddle {
namespace framework {

class CipherUtils {
 public:
  CipherUtils() = default;
  static std::string GenKey(int length);
  static std::string GenKeyToFile(int length, const std::string& filename);
  static std::string ReadKeyFromFile(const std::string& filename);

  static std::unordered_map<std::string, std::string> LoadConfig(
      const std::string& config_file);

  template <typename val_type>
  static bool GetValue(
      const std::unordered_map<std::string, std::string>& config,
      const std::string& key, val_type* output);

  static const int AES_DEFAULT_IV_SIZE;
  static const int AES_DEFAULT_TAG_SIZE;
};

template <>
bool CipherUtils::GetValue<bool>(
    const std::unordered_map<std::string, std::string>& config,
    const std::string& key, bool* output);

template <typename val_type>
bool CipherUtils::GetValue(
    const std::unordered_map<std::string, std::string>& config,
    const std::string& key, val_type* output) {
  auto itr = config.find(key);
  if (itr == config.end()) {
    return false;
  }
  std::istringstream iss(itr->second);
  iss >> *output;
  return true;
}

}  // namespace framework
}  // namespace paddle
