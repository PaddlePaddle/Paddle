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

#include "paddle/pten/core/lod_utils.h"

#include "paddle/pten/core/enforce.h"

namespace pten {

void AppendLoD(LoD *lod, const LoD &lod_length) {
  PADDLE_ENFORCE(
      lod->empty() || lod->size() == lod_length.size(),
      pten::errors::InvalidArgument(
          "The input LoD length should be equal to the appended LoD size, but "
          "received input LoD length is %d, actual LoD size is %d.",
          lod_length.size(),
          lod->size()));
  if (lod->empty()) {
    for (size_t i = 0; i < lod_length.size(); ++i) {
      lod->emplace_back(1, 0);  // size = 1, value = 0;
    }
    *lod = LoD(lod_length.size(), std::vector<size_t>({0}));
  }
  for (size_t i = 0; i < lod->size(); ++i) {
    auto &level = (*lod)[i];
    for (size_t len : lod_length[i]) {
      level.push_back(level.back() + len);
    }
  }
}

LoD ConvertToLengthBasedLoD(const LoD &offset_lod) {
  LoD length_lod;
  length_lod.reserve(offset_lod.size());
  for (size_t lvl = 0; lvl < offset_lod.size(); ++lvl) {
    std::vector<size_t> level;
    if (offset_lod[lvl].size() > 0) {
      level.reserve(offset_lod[lvl].size() - 1);
    }
    for (size_t idx = 0; idx < offset_lod[lvl].size() - 1; ++idx) {
      level.push_back(offset_lod[lvl][idx + 1] - offset_lod[lvl][idx]);
    }
    length_lod.push_back(level);
  }
  return length_lod;
}

}  // namespace pten
