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

#include <zlib.h>
#include <algorithm>
#include <memory>
#include <sstream>

#include "paddle/fluid/platform/enforce.h"
#include "snappystream.hpp"

namespace paddle {
namespace recordio {
constexpr size_t kMaxBufSize = 1024;

/**
 * Read Stream by a fixed sized buffer.
 * @param in input stream
 * @param limit read at most `limit` bytes from input stream. 0 means no limit
 * @param callback A function object with (const char* buf, size_t size) -> void
 * as its type.
 */
template <typename Callback>
static void ReadStreamByBuf(std::istream& in, size_t limit, Callback callback) {
  char buf[kMaxBufSize];
  std::streamsize actual_size;
  size_t counter = 0;
  size_t actual_max;
  while (!in.eof() ||
         (limit != 0 && counter >= limit)) {  // End of file or reach limit
    actual_max =
        limit != 0 ? std::min(limit - counter, kMaxBufSize) : kMaxBufSize;
    in.read(buf, actual_max);
    actual_size = in.gcount();
    if (actual_size == 0) {
      break;
    }
    callback(buf, actual_size);
    if (limit != 0) {
      counter += actual_size;
    }
  }
  in.clear();  // unset eof state
}

/**
 * Copy stream in to another stream
 */
static void PipeStream(std::istream& in, std::ostream& os) {
  ReadStreamByBuf(in, 0,
                  [&os](const char* buf, size_t len) { os.write(buf, len); });
}

/**
 * Calculate CRC32 from an input stream.
 */
static uint32_t Crc32Stream(std::istream& in, size_t limit = 0) {
  uint32_t crc = static_cast<uint32_t>(crc32(0, nullptr, 0));
  ReadStreamByBuf(in, limit, [&crc](const char* buf, size_t len) {
    crc = static_cast<uint32_t>(crc32(crc, reinterpret_cast<const Bytef*>(buf),
                                      static_cast<uInt>(len)));
  });
  return crc;
}

bool Chunk::Write(std::ostream& os, Compressor ct) const {
  // NOTE(dzhwinter): don't check records.numBytes instead, because
  // empty records are allowed.
  if (records_.empty()) {
    return false;
  }
  std::stringstream sout;
  std::unique_ptr<std::ostream> compressed_stream;
  switch (ct) {
    case Compressor::kNoCompress:
      break;
    case Compressor::kSnappy:
      compressed_stream.reset(new snappy::oSnappyStream(sout));
      break;
    default:
      PADDLE_THROW("Not implemented");
  }

  std::ostream& buf_stream = compressed_stream ? *compressed_stream : sout;

  for (auto& record : records_) {
    size_t sz = record.size();
    buf_stream.write(reinterpret_cast<const char*>(&sz), sizeof(uint32_t))
        .write(record.data(), record.size());
  }

  if (compressed_stream) {
    compressed_stream.reset();
  }

  sout.seekg(0, std::ios::end);
  uint32_t len = static_cast<uint32_t>(sout.tellg());
  sout.seekg(0, std::ios::beg);
  uint32_t crc = Crc32Stream(sout);
  Header hdr(static_cast<uint32_t>(records_.size()), crc, ct, len);
  hdr.Write(os);
  sout.seekg(0, std::ios::beg);
  sout.clear();
  PipeStream(sout, os);
  return true;
}

bool Chunk::Parse(std::istream& sin) {
  ChunkParser parser(sin);
  if (!parser.Init()) {
    return false;
  }
  Clear();
  while (parser.HasNext()) {
    Add(parser.Next());
  }
  return true;
}

ChunkParser::ChunkParser(std::istream& sin) : in_(sin) {}
bool ChunkParser::Init() {
  pos_ = 0;
  bool ok = header_.Parse(in_);
  if (!ok) {
    return ok;
  }
  auto beg_pos = in_.tellg();
  uint32_t crc = Crc32Stream(in_, header_.CompressSize());
  PADDLE_ENFORCE_EQ(header_.Checksum(), crc);
  in_.seekg(beg_pos, in_.beg);

  switch (header_.CompressType()) {
    case Compressor::kNoCompress:
      break;
    case Compressor::kSnappy:
      compressed_stream_.reset(new snappy::iSnappyStream(in_));
      break;
    default:
      PADDLE_THROW("Not implemented");
  }
  return true;
}

bool ChunkParser::HasNext() const { return pos_ < header_.NumRecords(); }

std::string ChunkParser::Next() {
  if (!HasNext()) {
    return "";
  }
  ++pos_;
  std::istream& stream = compressed_stream_ ? *compressed_stream_ : in_;
  uint32_t rec_len;
  stream.read(reinterpret_cast<char*>(&rec_len), sizeof(uint32_t));
  std::string buf;
  buf.resize(rec_len);
  stream.read(&buf[0], rec_len);
  PADDLE_ENFORCE_EQ(rec_len, stream.gcount());
  return buf;
}
}  // namespace recordio
}  // namespace paddle
