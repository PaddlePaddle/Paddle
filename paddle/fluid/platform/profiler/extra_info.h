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

#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/profiler/utils.h"

namespace paddle {
namespace platform {

class ExtraInfo {
 public:
  // Singleton.
  static ExtraInfo& GetInstance() {
    static ExtraInfo instance;
    return instance;
  }
  template <typename... Args>
  void AddMetaInfo(const std::string& key, const std::string& format,
                   Args... args);
  void Clear() { extra_info_.clear(); }
  std::unordered_map<std::string, std::string> GetMetaInfo() {
    return extra_info_;
  }

 private:
  ExtraInfo() {}
  std::unordered_map<std::string, std::string> extra_info_;
  DISABLE_COPY_AND_ASSIGN(ExtraInfo);
};

template <typename... Args>
void ExtraInfo::AddMetaInfo(const std::string& key, const std::string& format,
                            Args... args) {
  std::string value = string_format(format, args...);
  extra_info_[key] = value;
}

}  // namespace platform
}  // namespace paddle
