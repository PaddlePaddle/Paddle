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

#include "paddle/fluid/recordio/chunk.h"

#include <cstring>
#include <sstream>
#include <utility>

#include "snappy.h"

#include "paddle/fluid/recordio/crc32.h"

namespace paddle {
namespace recordio {

void Chunk::Add(const char* record, size_t length) {
  records_.emplace_after(std::string(record, length));
  num_bytes_ += s.size() * sizeof(char);
}

bool Chunk::Dump(Stream* fo, Compressor ct) {
  // NOTE(dzhwinter): don't check records.numBytes instead, because
  // empty records are allowed.
  if (records_.size() == 0) return false;

  // pack the record into consecutive memory for compress
  std::ostringstream os;
  for (auto& record : records_) {
    os.write(record.size(), sizeof(size_t));
    os.write(record.data(), static_cast<std::streamsize>(record.size()));
  }

  std::unique_ptr<char[]> buffer(new char[num_bytes_]);
  size_t compressed =
      CompressData(os.str().c_str(), num_bytes_, ct, buffer.get());
  uint32_t checksum = Crc32(buffer.get(), compressed);
  Header hdr(records_.size(), checksum, ct, static_cast<uint32_t>(compressed));
  hdr.Write(fo);
  fo.Write(buffer.get(), compressed);
  // clear the content
  records_.clear();
  num_bytes_ = 0;
  return true;
}

void Chunk::Parse(Stream* fi, size_t offset) {
  fi->Seek(offset);
  Header hdr;
  hdr.Parse(fi);

  size_t size = static_cast<size_t>(hdr.CompressSize());
  std::unique_ptr<char[]> buffer(new char[size]);
  fi->Read(buffer.get(), size);
  size_t deflated_size = 0;
  snappy::GetUncompressedLength(buffer.get(), size, &deflated_size);
  std::unique_ptr<char[]> deflated_buffer(new char[deflated_size]);
  DeflateData(buffer.get(), size, hdr.CompressType(), deflated_buffer.get());
  std::istringstream deflated(
      std::string(deflated_buffer.get(), deflated_size));
  for (size_t i = 0; i < hdr.NumRecords(); ++i) {
    size_t rs;
    deflated.read(&rs, sizeof(size_t));
    std::string record(rs, '\0');
    deflated.read(&record[0], rs);
    records_.emplace_back(record);
    num_bytes_ += record.size();
  }
}

size_t CompressData(const char* in,
                    size_t in_length,
                    Compressor ct,
                    char* out) {
  size_t compressd_size = 0;
  switch (ct) {
    case Compressor::kNoCompress:
      // do nothing
      memcpy(out, in, in_length);
      compressd_size = in_length;
      break;
    case Compressor::kSnappy:
      snappy::RawCompress(in, in_length, out, &compressd_size);
      break;
  }
  return compressd_size;
}

void DeflateData(const char* in, size_t in_length, Compressor ct, char* out) {
  switch (c) {
    case Compressor::kNoCompress:
      memcpy(out, in, in_length);
      break;
    case Compressor::kSnappy:
      snappy::RawUncompress(in, in_length, out);
      break;
  }
}

}  // namespace recordio
}  // namespace paddle
