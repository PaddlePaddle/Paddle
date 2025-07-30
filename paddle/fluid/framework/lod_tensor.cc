/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/lod_tensor.h"

#include <cstdint>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/version.h"

namespace paddle::framework {

std::string LegacyLoDToString(const LegacyLoD &lod) {
  std::ostringstream stream;
  stream << lod;
  return stream.str();
}

bool operator==(const LegacyLoD &a, const LegacyLoD &b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); i++) {
    const auto &a_level = a[i];
    const auto &b_level = b[i];
    if (a_level.size() != b_level.size()) {
      return false;
    }
    for (size_t j = 0; j < a_level.size(); j++) {
      if (a_level[j] != b_level[j]) {
        return false;
      }
    }
  }
  return true;
}

bool CheckLegacyLoD(const LegacyLoD &in, int tensor_height) {
  if (in.empty()) return true;
  for (const auto &level : in) {
    // check: there should be more than 2 offsets existing in each level.
    if (level.size() < 2) return false;
    // check: the first offset(the begin offset) of each level should be 0.
    if (level.front() != 0) return false;
    // check: all the offsets in a level should be non-descending
    if (!std::is_sorted(level.begin(), level.end())) {
      return false;
    }
  }
  // check: the lowest level's last offset should equals `tensor_height` if
  //        tensor_height>0.
  if (tensor_height > 0 &&
      static_cast<size_t>(tensor_height) != in.back().back())
    return false;

  // check: the higher level's last offset should equals the lower level's
  // size-1.
  // NOTE LegacyLoD store the levels from top to bottom, so the higher level
  // goes first.
  for (size_t level = 0; level < in.size() - 1; level++) {
    if (in[level].back() != in[level + 1].size() - 1) return false;
  }
  return true;
}

LegacyLoD ConvertToOffsetBasedLegacyLoD(const LegacyLoD &length_lod) {
  LegacyLoD offset_lod;
  offset_lod.reserve(length_lod.size());
  for (const auto &item : length_lod) {
    std::vector<size_t> level;
    level.reserve(item.size() + 1);
    size_t tmp = 0;
    level.push_back(tmp);
    for (auto i : item) {
      tmp += i;
      level.push_back(tmp);
    }
    offset_lod.push_back(level);
  }
  return offset_lod;
}

}  // namespace paddle::framework
