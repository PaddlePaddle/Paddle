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

#include "paddle/memory/memcpy.h"
#include "paddle/memory/memory.h"

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <iterator>

#include <glog/logging.h>

namespace paddle {
namespace framework {

std::ostream& operator<<(std::ostream& os, const LoD& lod) {
  os << "{";
  for (auto& v : lod) {
    os << "{";
    for (auto& i : v) {
      os << i << ",";
    }
    os << "}";
  }
  os << "}";

  return os;
}

LoD SliceLevels(const LoD& in, size_t level_begin, size_t level_end) {
  LoD new_lod;
  new_lod.reserve(level_end - level_begin);
  for (size_t i = level_begin; i < level_end; i++) {
    new_lod.emplace_back(in.at(i));
  }
  // transform the lowest level to absolute offset.
  LoD abs_offset_lod = ToAbsOffset(in);
  new_lod.back() = abs_offset_lod[level_end - 1];
  return new_lod;
}

LoD SliceInLevel(const LoD& in, size_t level, size_t elem_begin,
                 size_t elem_end) {
  PADDLE_ENFORCE_LT(level, in.size());
  PADDLE_ENFORCE_LT(elem_end, in[level].size());

  LoD res;
  res.resize(in.size() - level);
  // copy the first level
  res[0].assign(in[level].begin() + elem_begin,
                in[level].begin() + elem_end + 1);
  for (size_t lvl = 1; lvl < res.size(); lvl++) {
    const auto& in_level = in[level + lvl];
    const auto& above_level = res[lvl - 1];
    auto& out_level = res[lvl];
    out_level.assign(in_level.begin() + above_level.front(),
                     in_level.begin() + above_level.back() + 1);
  }
  for (size_t lvl = 0; lvl < res.size(); lvl++) {
    // to make the first offset equals 0, all the elements minus the first
    // element
    size_t front = res[lvl].front();
    for (auto& ele : res[lvl]) {
      ele -= front;
    }
  }
  return res;
}

LoD ToAbsOffset(const LoD& in) {
  // the lowest level stores relative offsets
  if (in.empty() || in.size() == 1) return in;
  LoD result = in;
  for (int level = result.size() - 2; level >= 0; level--) {
    for (auto& ele : result[level]) {
      ele = result[level + 1][ele];
    }
  }
  return result;
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

size_t LoDTensor::NumElements(size_t level, size_t idx) const {
  PADDLE_ENFORCE_LT(level, NumLevels());
  PADDLE_ENFORCE_LT(idx, NumElements(level));
  return lod_[level][idx + 1] - lod_[level][idx];
}

size_t LoDTensor::NumInstancesInElement(size_t level, size_t idx) const {
  PADDLE_ENFORCE_LT(level, NumLevels());
  PADDLE_ENFORCE_LT(idx, NumElements(level));
  auto abs_lod = ToAbsOffset(lod());
  size_t begin = abs_lod[level][idx];
  size_t end = abs_lod[level][idx + 1];
  return end - begin;
}

void LoDTensor::ShrinkLevels(size_t level_begin, size_t level_end) {
  auto new_lod = framework::SliceLevels(lod_, level_begin, level_end);
  lod_ = new_lod;
}

void LoDTensor::ShrinkInLevel(size_t level, size_t elem_begin,
                              size_t elem_end) {
  PADDLE_ENFORCE_LT(level, NumLevels());
  PADDLE_ENFORCE_LT(elem_begin, NumElements(level));
  PADDLE_ENFORCE_LT(elem_end, NumElements(level) + 1);

  auto abs_lod = framework::ToAbsOffset(lod());
  auto new_lod = framework::SliceInLevel(lod_, level, elem_begin, elem_end);
  lod_ = new_lod;

  // slice the underlying tensor
  size_t begin = abs_lod[level][elem_begin];
  size_t end = abs_lod[level][elem_end];
  PADDLE_ENFORCE_LT(begin, end, "Cannot shrink, the result tensor is empty.");
  ShareDataWith(Slice(begin, end));
}

using LoDAndOffset = std::pair<LoD, std::pair<size_t, size_t>>;
LoDAndOffset GetSubLoDAndAbsoluteOffset(const LoD& lod, size_t start_idx,
                                        size_t end_idx, size_t start_level) {
  LoD sub_lod;

  for (size_t level_idx = start_level; level_idx < lod.size(); ++level_idx) {
    PADDLE_ENFORCE_LE(start_idx, end_idx);
    PADDLE_ENFORCE_LT(end_idx, lod[level_idx].size());
    std::vector<size_t> level_lens;
    for (size_t i = start_idx; i < end_idx; ++i) {
      level_lens.push_back(lod[level_idx][i + 1] - lod[level_idx][i]);
    }
    sub_lod.emplace_back(level_lens);
    start_idx = lod[level_idx][start_idx];
    end_idx = lod[level_idx][end_idx];
  }

  return LoDAndOffset{sub_lod, {start_idx, end_idx}};
}

void AppendLoD(LoD* lod, const LoD& lod_length) {
  PADDLE_ENFORCE(
      lod->empty() || lod->size() == lod_length.size(),
      "The lod_length should has the same size with the appended lod.");
  if (lod->empty()) {
    *lod = LoD(lod_length.size(), std::vector<size_t>({0}));
  }
  for (size_t i = 0; i < lod->size(); ++i) {
    auto& level = (*lod)[i];
    for (size_t len : lod_length[i]) {
      level.push_back(level.back() + len);
    }
  }
}

}  // namespace framework
}  // namespace paddle
