// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/debug_manager.h"

namespace cinn {
namespace common {

inline std::vector<std::pair<std::string, absl::any>> &GetVec(
    absl::any &data) {  // NOLINT
  return absl::any_cast<std::vector<std::pair<std::string, absl::any>> &>(data);
}

//! AppendTypeSuffix for multiple types.
// @{
template <>
inline std::string DebugManager::AppendTypeSuffix<int32_t>(
    const std::string &key) {
  return key + "_i32";
}
template <>
inline std::string DebugManager::AppendTypeSuffix<int64_t>(
    const std::string &key) {
  return key + "_i64";
}
template <>
inline std::string DebugManager::AppendTypeSuffix<float>(
    const std::string &key) {
  return key + "_f32";
}
template <>
inline std::string DebugManager::AppendTypeSuffix<double>(
    const std::string &key) {
  return key + "_f64";
}
template <>
inline std::string DebugManager::AppendTypeSuffix<bool>(
    const std::string &key) {
  return key + "_b";
}
template <>
inline std::string DebugManager::AppendTypeSuffix<std::string>(
    const std::string &key) {
  return key + "_s";
}
// @}

void DebugManager::Append(const std::string &key, absl::any value) {
  GetVec(data_).push_back(std::make_pair(key, value));
}
void DebugManager::Append(const std::string &key, int32_t value) {
  GetVec(data_).push_back(
      std::make_pair(AppendTypeSuffix<int32_t>(key), value));
}
void DebugManager::Append(const std::string &key, bool value) {
  GetVec(data_).push_back(std::make_pair(AppendTypeSuffix<bool>(key), value));
}
void DebugManager::Append(const std::string &key, const std::string &value) {
  GetVec(data_).push_back(
      std::make_pair(AppendTypeSuffix<std::string>(key), value));
}

void DebugManager::Clear() { GetVec(data_).clear(); }

}  // namespace common
}  // namespace cinn
