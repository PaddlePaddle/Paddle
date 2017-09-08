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

#include "paddle/framework/lod_tensor.h"

#include <glog/logging.h>

namespace paddle {
namespace framework {

LoD SliceLevels(const LoD& in, size_t level_begin, size_t level_end) {
  LoD new_lod;
  new_lod.reserve(level_end - level_begin);
  for (size_t i = level_begin; i < level_end; i++) {
    new_lod.emplace_back(in.at(i));
  }
  return new_lod;
}

LoD SliceInLevel(const LoD& in, size_t level, size_t elem_begin,
                 size_t elem_end) {
  // slice the lod.
  LoD new_lod;
  new_lod.reserve(in.size() - level);
  auto start = in.at(level)[elem_begin];
  auto end = in.at(level)[elem_end];

  for (auto it = in.begin() + level; it != in.end(); it++) {
    auto it_begin = std::find(it->begin(), it->end(), start);
    auto it_end = std::find(it_begin, it->end(), end);
    PADDLE_ENFORCE(it_begin != it->end(), "error in parsing lod info");
    PADDLE_ENFORCE(it_end != it->end(), "error in parsing lod info");
    new_lod.emplace_back(it_begin, it_end + 1);
    // reset offset if tensor is copyed and sliced.
    std::transform(new_lod.back().begin(), new_lod.back().end(),
                   new_lod.back().begin(),
                   [start](int v) { return v - start; });
    PADDLE_ENFORCE_EQ(new_lod.back().front(), 0, "error in slice LoD");
  }
  PADDLE_ENFORCE_LE(new_lod.size(), in.size());
  return new_lod;
}

bool operator==(const LoD& a, const LoD& b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); i++) {
    const auto& a_level = a[i];
    const auto& b_level = b[i];
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

void LoDTensor::SliceLevels(size_t level_begin, size_t level_end) {
  auto new_lod = framework::SliceLevels(lod_, level_begin, level_end);
  lod_ = new_lod;
}

void LoDTensor::SliceInLevel(size_t level, size_t elem_begin, size_t elem_end) {
  PADDLE_ENFORCE(level < NumLevels(), "level [%d] out of range [%d]", level,
                 NumLevels());
  PADDLE_ENFORCE(elem_begin < NumElements(level),
                 "element begin [%d] out of range [%d]", elem_begin,
                 NumElements(level));
  PADDLE_ENFORCE(elem_end < NumElements(level) + 1,
                 "element end [%d] out of range [%d]", elem_end,
                 NumElements(level));

  auto new_lod = framework::SliceInLevel(lod_, level, elem_begin, elem_end);
  lod_ = new_lod;
}

}  // namespace framework
}  // namespace paddle
