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

#pragma once
#include <absl/types/any.h>

#include <string>
#include <utility>
#include <vector>

namespace cinn {
namespace common {

/**
 * Container for debug info.
 * DebugManager is integrated into the global Context, and used to log
 * something(but not print to stdout directly).
 */
class DebugManager {
 public:
  void Append(const std::string& key, int32_t value);
  void Append(const std::string& key, bool value);
  void Append(const std::string& key, const std::string& value);
  void Clear();

 protected:
  void Append(const std::string& key, absl::any value);

  template <typename T>
  inline std::string AppendTypeSuffix(const std::string& key) {
    return key;
  }

 private:
  //! hide the type of vector<pair<string, any>>
  absl::any data_;
};

}  // namespace common
}  // namespace cinn
