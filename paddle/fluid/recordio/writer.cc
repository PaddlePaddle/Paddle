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

#include "paddle/fluid/recordio/writer.h"

namespace paddle {
namespace recordio {

Writer::Writer(Stream* fo) : stream_(fo), max_chunk_size_(0), compressor_(0) {}

Writer::Writer(Stream* fo, int maxChunkSize, int compressor)
    : stream_(fo),
      max_chunk_size_(maxChunkSize),
      compressor_(static_cast<Compressor>(compressor)) {
  chunk_.reset(new Chunk);
}

size_t Writer::Write(const char* buf, size_t length) {
  if (stream_ == nullptr) {
    LOG(WARNING) << "Cannot write since writer had been closed.";
    return 0;
  }
  if ((length + chunk_->NumBytes()) > max_chunk_size_) {
    chunk_->Dump(stream_, compressor_);
  }
  chunk_->Add(buf, length);
  return length;
}

// size_t Writer::Write(const char* buf, size_t length) {
//   return Write(std::string(buf, length));
// }

// size_t Writer::Write(std::string&& buf) {}

void Writer::Close() {
  chunk_->Dump(stream_, compressor_);
  stream_ = nullptr;
}

}  // namespace recordio
}  // namespace paddle
