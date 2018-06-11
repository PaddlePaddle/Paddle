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

#include "paddle/fluid/framework/lod_rank_table.h"

namespace paddle {
namespace framework {
void LoDRankTable::Reset(const LoD& lod, size_t level) {
  this->coarse_lod_.clear();
  this->items_.clear();
  PADDLE_ENFORCE(level < lod.size(),
                 "Cannot rank lod since the level %d is less than lod size %d",
                 level, lod.size());
  coarse_lod_.reserve(level);
  for (size_t i = 0; i < level; ++i) {
    coarse_lod_.push_back(lod[i]);
  }
  auto& vec = lod[level];
  for (size_t i = 0; i < vec.size() - 1; ++i) {
    TableItem item;
    item.index = i;
    item.length = vec[i + 1] - vec[i];
    VLOG(10) << "Add item to rank table " << item.index << " " << item.length;
    items_.emplace_back(item);
  }
  // NOTE(yuyang18):
  //
  // The time complexity of stable_sort is O(N*log(N)) if additional memory is
  // available. It is easy to debug and unit test when using `stable_sort`
  // instead of `sort`. Also, the items of a rank table will not be too large.
  std::stable_sort(items_.begin(), items_.end(),
                   [](const TableItem& a, const TableItem& b) {
                     return a.length > b.length;
                   });
}

}  // namespace framework

std::ostream& operator<<(std::ostream& out,
                         const framework::LoDRankTable& table) {
  out << "NumOfSequence " << table.items().size() << "\n";
  for (auto& each_item : table.items()) {
    out << "\tSeq #" << each_item.index << ", Len=" << each_item.length << "\n";
  }
  return out;
}
}  // namespace paddle
