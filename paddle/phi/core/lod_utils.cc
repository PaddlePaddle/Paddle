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

#include "paddle/phi/core/lod_utils.h"

#include "paddle/phi/core/enforce.h"

namespace phi {

LegacyLoD ToAbsOffset(const LegacyLoD &in) {
  // the lowest level stores relative offsets
  if (in.empty() || in.size() == 1) return in;
  LegacyLoD result = in;
  for (auto level = static_cast<int>(in.size() - 2); level >= 0; level--) {
    for (size_t i = 0; i < in[level].size(); ++i) {
      size_t index = in[level][i];
      result[level][i] = result[level + 1][index];
    }
  }
  return result;
}

void AppendLegacyLoD(LegacyLoD *lod, const LegacyLoD &lod_length) {
  PADDLE_ENFORCE_EQ(
      (lod->empty() || lod->size() == lod_length.size()),
      true,
      common::errors::InvalidArgument(
          "The input LegacyLoD length should be equal to the appended "
          "LegacyLoD size, but "
          "received input LegacyLoD length is %d, actual LegacyLoD size is %d.",
          lod_length.size(),
          lod->size()));
  if (lod->empty()) {
    for (size_t i = 0; i < lod_length.size(); ++i) {
      lod->emplace_back(1, 0);  // size = 1, value = 0;
    }
    *lod = LegacyLoD(lod_length.size(), std::vector<size_t>({0}));
  }
  for (size_t i = 0; i < lod->size(); ++i) {
    auto &level = (*lod)[i];
    for (size_t len : lod_length[i]) {
      level.push_back(level.back() + len);
    }
  }
}

LegacyLoD ConvertToLengthBasedLegacyLoD(const LegacyLoD &offset_lod) {
  LegacyLoD length_lod;
  length_lod.reserve(offset_lod.size());
  for (const auto &item : offset_lod) {
    std::vector<size_t> level;
    if (!item.empty()) {
      level.reserve(item.size() - 1);
    }
    for (size_t idx = 0; idx < item.size() - 1; ++idx) {
      level.push_back(item[idx + 1] - item[idx]);
    }
    length_lod.push_back(level);
  }
  return length_lod;
}

}  // namespace phi
