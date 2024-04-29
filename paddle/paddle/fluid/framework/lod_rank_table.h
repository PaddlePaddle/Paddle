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

#pragma once
#include <iosfwd>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {

// LoD Rank Table stores the `level` of `lod` which is ordered by sequence
// length in descending order. It is useful when implement dynamic RNN and is
// shared by dynamic RNN memory, dynamic RNN slice input and dynamic RNN slice
// output operators.
//
// The table item contains two element. The length of sequence and the index of
// sequence in that level.
//
// LoDRankTable also stores the coarse_lod, which is the lod information whose
// level is less than input level, in order to restore the output LoD
// information.
class LoDRankTable {
 public:
  struct TableItem {
    size_t index;
    size_t length;
  };

  LoDRankTable() {}

  void Reset(const LoD& lod, size_t level);

  const std::vector<TableItem>& items() const { return this->items_; }

  const LoD& coarse_lod() const { return this->coarse_lod_; }

  size_t level() const { return coarse_lod_.size(); }

 private:
  LoD coarse_lod_;
  std::vector<TableItem> items_;
};

}  // namespace framework

std::ostream& operator<<(std::ostream& out,
                         const framework::LoDRankTable& table);

}  // namespace paddle
