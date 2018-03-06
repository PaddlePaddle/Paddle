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

void Index::LoadIndex(FileStream* fi) {
  int64_t offset = 0;
  while (!fi->Eof()) {
    Header hdr;
    hdr.Parse(fi);
    chunk_offsets_.push_back(offset);
    chunk_lens_.push_back(hdr.NumRecords());
    chunk_records_.push_back(hdr.NumRecords());
    num_records_ += hdr.NumRecords();
    offset += hdr.CompressSize();
  }
}

Index Index::ChunkIndex(int i) { Index idx; }

std::pair<int, int> Index::Locate(int record_idx) {
  std::pair<int, int> range(-1, -1);
  int sum = 0;
  for (size_t i = 0; i < chunk_lens_.size(); ++i) {
    int len = static_cast<int>(chunk_lens_[i]);
    sum += len;
    if (record_idx < sum) {
      range.first = static_cast<int>(i);
      range.second = record_idx - sum + len;
    }
  }
  return range;
}

RangeScanner::RangeScanner(Stream* fi, Index idx, int start, int len)
    : stream_(fi), index_(idx) {
  if (start < 0) {
    start = 0;
  }
  if (len < 0 || start + len >= idx.NumRecords()) {
    len = idx.NumRecords() - start;
  }

  start_ = start;
  end_ = start + len;
  cur_ = start - 1;  // The intial status required by Scan
  chunk_index_ = -1;
  chunk_.reset(new Chunk);
}

bool RangeScanner::Scan() {
  ++cur_;
  if (cur_ >= end_) {
    return false;
  } else {
    auto cursor = index_.Locate(cur_);
    if (chunk_index_ != cursor.first) {
      chunk_index_ = cursor.first;
      chunk_->Parse(fi, index_.ChunkOffsets[chunk_index_]);
    }
  }
  return true;
}

const std::string RangeScanner::Record() {
  auto cursor = index_.Locate(cur_);
  return chunk_->Record(cursor.second);
}

}  // namespace recordio
}  // namespace paddle
