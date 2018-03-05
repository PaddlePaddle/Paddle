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

#pragma once

#include "paddle/fluid/recordio/io.h"

namespace paddle {
namespace recordio {

// Index consists offsets and sizes of the consequetive chunks in a RecordIO
// file.
//
// Index supports Gob. Every field in the Index needs to be exported
// for the correct encoding and decoding using Gob.
class Index {
public:
  int NumRecords() { return num_records_; }
  // NumChunks returns the total number of chunks in a RecordIO file.
  int NumChunks() { return chunk_lens_.size(); }
  // ChunkIndex return the Index of i-th Chunk.
  int ChunkIndex(int i);

  // Locate returns the index of chunk that contains the given record,
  // and the record index within the chunk.  It returns (-1, -1) if the
  // record is out of range.
  void Locate(int record_idx, std::pair<int, int>* out) {
    size_t sum = 0;
    for (size_t i = 0; i < chunk_lens_.size(); ++i) {
      sum += chunk_lens_[i];
      if (static_cast<size_t>(record_idx) < sum) {
        out->first = i;
        out->second = record_idx - sum + chunk_lens_[i];
        return;
      }
    }
    // out->swap(std::make_pair<int,int>(-1, -1));
    out->first = -1;
    out->second = -1;
  }

private:
  // the offset of each chunk in a file.
  std::vector<int64_t> chunk_offsets_;
  // the length of each chunk in a file.
  std::vector<uint32_t> chunk_lens_;
  // the numer of all records in a file.
  int num_records_;
  // the number of records in chunks.
  std::vector<int> chunk_records_;
};

// RangeScanner
// creates a scanner that sequencially reads records in the
// range [start, start+len).  If start < 0, it scans from the
// beginning.  If len < 0, it scans till the end of file.
class RangeScanner {
public:
  RangeScanner(Stream* fi, Index idx, int start, int end);
  bool Scan();
  const std::string Record();

private:
  Stream* fi;
  Index index_;
  int start_, end_, cur_;
  int chunk_index_;
  std::unique_ptr<Chunk> chunk_;
};

}  // namespace recordio
}  // namespace paddle
