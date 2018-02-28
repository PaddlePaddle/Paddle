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

#include "paddle/fluid/platform/macros.h"  // for DISABLE COPY ASSIGN
#include "paddle/fluid/recordio/header.h"

namespace paddle {
namespace recordio {

// Writer creates a RecordIO file.
class Writer {
public:
  Writer(std::ostream& os);
  Writer(std::ostream& os, int maxChunkSize, int c);

  // Writes a record.  It returns an error if Close has been called.
  size_t Write(const char* buf, size_t length);
  size_t Write(const std::string& buf);
  size_t Write(std::string&& buf);

  // Close flushes the current chunk and makes the writer invalid.
  void Close();

private:
  // Set rdstate to mark a closed writer
  std::ostream stream_;
  std::unique_ptr<Chunk> chunk_;
  // total records size, excluding metadata, before compression.
  int max_chunk_size_;
  int compressor_;
  DISABLE_COPY_AND_ASSIGN(Writer);
};

template <typename T>
Writer& operator<<(const T& val) {
  stream_ << val;
  return *this;
}

}  // namespace recordio
}  // namespace paddle
