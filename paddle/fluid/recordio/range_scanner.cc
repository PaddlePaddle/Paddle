//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/recordio/range_scanner.h"

namespace paddle {
namespace recordio {

Index Index::ChunkIndex(int i) { Index idx; }

RangeScanner::RangeScanner(std::istream is, Index idx, int start, int len)
    : stream_(is.rdbuf()), index_(idx) {
  if (start < 0) {
    start = 0;
  }
  if (len < 0 || start + len >= idx.NumRecords()) {
    len = idx.NumRecords() - start;
  }

  start_ = start;
  end_ = start + len;
  cur_ = start - 1;
  chunk_index_ = -1;
  // chunk_->reset(new Chunk());
}

bool RangeScanner::Scan() {}

const std::string RangeScanner::Record() {
  // int i = index_.Locate(cur_);
  // return chunk_->Record(i);
}

}  // namespace recordio
}  // namespace paddle
