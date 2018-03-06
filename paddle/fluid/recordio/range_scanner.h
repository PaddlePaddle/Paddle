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

#include <utility>

#include "paddle/fluid/recordio/chunk.h"
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
  Index() : num_records_(0) {}
  // LoadIndex scans the file and parse chunkOffsets, chunkLens, and len.
  void LoadIndex(Stream* fi);
  // NumRecords returns the total number of all records in a RecordIO file.
  int NumRecords() { return num_records_; }
  // NumChunks returns the total number of chunks in a RecordIO file.
  int NumChunks() { return chunk_lens_.size(); }
  // ChunkIndex return the Index of i-th Chunk.
  int ChunkIndex(int i);

  int64_t ChunkOffsets(int i) { return chunk_offsets_[i]; }

  // Locate returns the index of chunk that contains the given record,
  // and the record index within the chunk.  It returns (-1, -1) if the
  // record is out of range.
  std::pair<int, int> Locate(int record_idx);

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
class RangeScanner {
public:
  // creates a scanner that sequencially reads records in the
  // range [start, start+len).  If start < 0, it scans from the
  // beginning.  If len < 0, it scans till the end of file.
  RangeScanner(Stream* fi, Index idx, int start, int end);
  // Scan moves the cursor forward for one record and loads the chunk
  // containing the record if not yet.
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
