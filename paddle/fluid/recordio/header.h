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

#include <sstream>

namespace paddle {
namespace recordio {

// MagicNumber for memory checking
constexpr uint32_t kMagicNumber = 0x01020304;

enum class Compressor : uint32_t {
  // NoCompression means writing raw chunk data into files.
  // With other choices, chunks are compressed before written.
  kNoCompress = 0,
  // Snappy had been the default compressing algorithm widely
  // used in Google.  It compromises between speech and
  // compression ratio.
  kSnappy = 1,
  // Gzip is a well-known compression algorithm.  It is
  // recommmended only you are looking for compression ratio.
  kGzip = 2,
};

// Header is the metadata of Chunk
class Header {
 public:
  Header();
  Header(uint32_t num, uint32_t sum, Compressor ct, uint32_t cs);

  void Write(std::ostream& os) const;

  // returns true if OK, false if eof
  bool Parse(std::istream& is);

  uint32_t NumRecords() const { return num_records_; }
  uint32_t Checksum() const { return checksum_; }
  Compressor CompressType() const { return compressor_; }
  uint32_t CompressSize() const { return compress_size_; }

 private:
  uint32_t num_records_;
  uint32_t checksum_;
  Compressor compressor_;
  uint32_t compress_size_;
};

// Allow Header Loggable
std::ostream& operator<<(std::ostream& os, Header h);
bool operator==(Header l, Header r);

}  // namespace recordio
}  // namespace paddle
