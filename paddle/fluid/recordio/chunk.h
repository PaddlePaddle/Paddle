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
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/recordio/header.h"

namespace paddle {
namespace recordio {

// A Chunk contains the Header and optionally compressed records.
class Chunk {
 public:
  Chunk() : num_bytes_(0) {}
  void Add(const std::string& buf) {
    num_bytes_ += buf.size();
    records_.emplace_back(buf);
  }
  // dump the chunk into w, and clears the chunk and makes it ready for
  // the next add invocation.
  bool Write(std::ostream& fo, Compressor ct) const;
  void Clear() {
    records_.clear();
    num_bytes_ = 0;
  }

  // returns true if ok, false if eof
  bool Parse(std::istream& sin);
  size_t NumBytes() const { return num_bytes_; }
  size_t NumRecords() const { return records_.size(); }
  const std::string& Record(int i) const { return records_[i]; }

  bool Empty() const { return records_.empty(); }

 private:
  std::vector<std::string> records_;
  // sum of record lengths in bytes.
  size_t num_bytes_;
  DISABLE_COPY_AND_ASSIGN(Chunk);
};

class ChunkParser {
 public:
  explicit ChunkParser(std::istream& sin);

  bool Init();
  std::string Next();
  bool HasNext() const;

 private:
  Header header_;
  uint32_t pos_{0};
  std::istream& in_;
  std::unique_ptr<std::istream> compressed_stream_;
};

}  // namespace recordio
}  // namespace paddle
