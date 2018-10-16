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

#include <string>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace recordio {

Header::Header()
    : num_records_(0),
      checksum_(0),
      compressor_(Compressor::kNoCompress),
      compress_size_(0) {}

Header::Header(uint32_t num, uint32_t sum, Compressor c, uint32_t cs)
    : num_records_(num), checksum_(sum), compressor_(c), compress_size_(cs) {}

bool Header::Parse(std::istream& is) {
  uint32_t magic;
  is.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
  size_t read_size = is.gcount();
  if (read_size < sizeof(uint32_t)) {
    return false;
  }
  PADDLE_ENFORCE_EQ(magic, kMagicNumber);

  is.read(reinterpret_cast<char*>(&num_records_), sizeof(uint32_t))
      .read(reinterpret_cast<char*>(&checksum_), sizeof(uint32_t))
      .read(reinterpret_cast<char*>(&compressor_), sizeof(uint32_t))
      .read(reinterpret_cast<char*>(&compress_size_), sizeof(uint32_t));
  return true;
}

void Header::Write(std::ostream& os) const {
  os.write(reinterpret_cast<const char*>(&kMagicNumber), sizeof(uint32_t))
      .write(reinterpret_cast<const char*>(&num_records_), sizeof(uint32_t))
      .write(reinterpret_cast<const char*>(&checksum_), sizeof(uint32_t))
      .write(reinterpret_cast<const char*>(&compressor_), sizeof(uint32_t))
      .write(reinterpret_cast<const char*>(&compress_size_), sizeof(uint32_t));
}

std::ostream& operator<<(std::ostream& os, Header h) {
  os << "Header: " << h.NumRecords() << ", " << h.Checksum() << ", "
     << static_cast<uint32_t>(h.CompressType()) << ", " << h.CompressSize();
  return os;
}

bool operator==(Header l, Header r) {
  return l.NumRecords() == r.NumRecords() && l.Checksum() == r.Checksum() &&
         l.CompressType() == r.CompressType() &&
         l.CompressSize() == r.CompressSize();
}

}  // namespace recordio
}  // namespace paddle
