// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/phi/core/platform/profiler/utils.h"

namespace paddle {
namespace platform {

class ExtraInfo {
 public:
  ExtraInfo() {}
  template <typename... Args>
  void AddExtraInfo(const std::string& key,
                    const std::string& format,
                    Args... args);
  void Clear() { extra_info_.clear(); }
  std::unordered_map<std::string, std::string> GetExtraInfo() {
    return extra_info_;
  }

 private:
  std::unordered_map<std::string, std::string> extra_info_;
};

template <typename... Args>
void ExtraInfo::AddExtraInfo(const std::string& key,
                             const std::string& format,
                             Args... args) {
  std::string value = string_format(format, args...);
  extra_info_[key] = value;
}

}  // namespace platform
}  // namespace paddle
