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

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Chunk
// a chunk contains the Header and optionally compressed records.
class Chunk {
public:
  Chunk() = default;
  void Add(const char* record, size_t length);
  void Add(const std::string&);

  bool Dump(std::ostream& os, Compressor ct);
  void Parse(std::istream& iss, int64_t offset);
  const std::string Record(int i) { return records_[i]; }

private:
  std::vector<std::string> records_;
  size_t num_bytes_;
};

size_t CompressData(const std::stringstream& ss, Compressor ct, char* buffer);

uint32_t DeflateData(char* buffer, uint32_t size, Compressor c);

// implementation
void Chunk::Add(const std::string& s) {
  num_bytes_ += s.size() * sizeof(char);
  records_.emplace_back(std::move(s));
  // records_.resize(records_.size()+1);
  // records_[records_.size()-1] = s;
}

void Chunk::Add(const char* record, size_t length) {
  Add(std::string(record, length));
}

bool Chunk::Dump(std::ostream& os, Compressor ct) {
  if (records_.size() == 0) return false;

  // TODO(dzhwinter):
  // we pack the string with same size buffer,
  // then compress with another buffer.
  // Here can be optimized if it is the bottle-neck.
  std::ostringstream oss;
  for (auto& record : records_) {
    unsigned len = record.size();
    oss << len;
    oss << record;
    // os.write(std::to_string(len).c_str(), sizeof(unsigned));
    // os.write(record.c_str(), record.size());
  }
  std::unique_ptr<char[]> buffer(new char[kDefaultMaxChunkSize]);
  size_t compressed = CompressData(oss.str(), ct, buffer.get());

  // TODO(dzhwinter): crc32 checksum
  size_t checksum = compressed;

  Header hdr(records_.size(), checksum, ct, compressed);

  return true;
}

void Chunk::Parse(std::istream& iss, int64_t offset) {
  iss.seekg(offset, iss.beg);
  Header hdr;
  hdr.Parse(iss);

  std::unique_ptr<char[]> buffer(new char[kDefaultMaxChunkSize]);
  iss.read(buffer.get(), static_cast<size_t>(hdr.CompressSize()));
  // TODO(dzhwinter): checksum
  uint32_t deflated_size =
      DeflateData(buffer.get(), hdr.CompressSize(), hdr.CompressType());
  std::istringstream deflated(std::string(buffer.get(), deflated_size));
  for (size_t i = 0; i < hdr.NumRecords(); ++i) {
    uint32_t rs;
    deflated >> rs;
    std::string record(rs, '\0');
    deflated.read(&record[0], rs);
    records_.emplace_back(record);
    num_bytes_ += record.size();
  }
}

uint32_t DeflateData(char* buffer, uint32_t size, Compressor c) {
  uint32_t deflated_size = 0;
  std::string uncompressed;
  switch (c) {
    case Compressor::kNoCompress:
      deflated_size = size;
      break;
    case Compressor::kSnappy:
      // snappy::Uncompress(buffer, size, &uncompressed);
      // deflated_size = uncompressed.size();
      // memcpy(buffer, uncompressed.data(), uncompressed.size() *
      // sizeof(char));
      break;
  }
  return deflated_size;
}
