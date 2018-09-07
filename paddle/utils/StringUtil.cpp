/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "StringUtil.h"

namespace paddle {
namespace str {

bool endsWith(const std::string& str, const std::string& ext) {
  if (str.size() >= ext.size() && ext == str.substr(str.size() - ext.size())) {
    return true;
  } else {
    return false;
  }
}

void split(const std::string& str, char sep, std::vector<std::string>* pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

bool startsWith(const std::string& str, const std::string& prefix) {
  if (prefix.size() <= str.size()) {
    for (size_t i = 0; i < prefix.size(); ++i) {
      if (str[i] != prefix[i]) return false;
    }
    return true;
  } else {
    return false;
  }
}

}  // namespace str
}  // namespace paddle
