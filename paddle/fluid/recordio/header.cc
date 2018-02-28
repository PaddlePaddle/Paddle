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

#include "paddle/fluid/recordio/header.h"

namespace paddle {
namespace recordio {

Header::Header()
    : num_records_(0),
      checksum_(0),
      compressor_(Compressor::kNoCompress),
      compress_size_(0) {}

Header::Header(uint32_t num, uint32_t sum, Compressor c, uint32_t cs)
    : num_records_(num), checksum_(sum), compressor_(c), compress_size_(cs) {}

void Header::Parse(std::istream& iss) {
  iss.read(reinterpret_cast<char*>(&num_records_), sizeof(uint32_t));
  iss.read(reinterpret_cast<char*>(&checksum_), sizeof(uint32_t));
  iss.read(reinterpret_cast<char*>(&compressor_), sizeof(uint32_t));
  iss.read(reinterpret_cast<char*>(&compress_size_), sizeof(uint32_t));
}

void Header::Write(std::ostream& os) {
  os.write(reinterpret_cast<char*>(&num_records_), sizeof(uint32_t));
  os.write(reinterpret_cast<char*>(&checksum_), sizeof(uint32_t));
  os.write(reinterpret_cast<char*>(&compressor_), sizeof(uint32_t));
  os.write(reinterpret_cast<char*>(&compress_size_), sizeof(uint32_t));
}

// std::ostream& operator << (std::ostream& os, Header h) {
//   os << h.num_records_
//      << h.checksum_
//      << static_cast<uint32_t>(h.compressor_)
//      << h.compress_size_;
//   return os;
// }

std::ostream& operator<<(std::ostream& os, Header h) {
  os << h.NumRecords() << h.Checksum()
     << static_cast<uint32_t>(h.CompressType()) << h.CompressSize();
  return os;
}

// bool operator==(Header l, Header r) {
//   return num_records_ == rhs.NumRecords() &&
//     checksum_ == rhs.Checksum() &&
//     compressor_ == rhs.CompressType() &&
//     compress_size_ == rhs.CompressSize();
// }

bool operator==(Header l, Header r) {
  return l.NumRecords() == r.NumRecords() && l.Checksum() == r.Checksum() &&
         l.CompressType() == r.CompressType() &&
         l.CompressSize() == r.CompressSize();
}

// size_t CompressData(const std::string& os, Compressor ct, char* buffer) {
//   size_t compress_size = 0;

//   // std::unique_ptr<char[]> buffer(new char[kDefaultMaxChunkSize]);
//   // std::string compressed;
//   compress_size =os.size();
//   memcpy(buffer, os.c_str(), compress_size);
//   return compress_size;
// }

}  // namespace recordio
}  // namespace paddle
