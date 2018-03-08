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

#include <memory>
#include <sstream>
#include "paddle/fluid/platform/enforce.h"
#include "snappystream.hpp"
#include "zlib.h"

namespace paddle {
namespace recordio {
constexpr size_t kMaxBufSize = 1024;

template <typename Callback>
static void ReadStreamByBuf(std::istream& in, int limit, Callback callback) {
  char buf[kMaxBufSize];
  std::streamsize actual_size;
  size_t counter = 0;
  do {
    auto actual_max =
        limit > 0 ? std::min(limit - counter, kMaxBufSize) : kMaxBufSize;
    actual_size = in.readsome(buf, actual_max);
    if (actual_size == 0) {
      break;
    }
    callback(buf, actual_size);
    if (limit > 0) {
      counter += actual_size;
    }
  } while (actual_size == kMaxBufSize);
}

static void PipeStream(std::istream& in, std::ostream& os) {
  ReadStreamByBuf(
      in, -1, [&os](const char* buf, size_t len) { os.write(buf, len); });
}
static uint32_t Crc32Stream(std::istream& in, int limit = -1) {
  auto crc = crc32(0, nullptr, 0);
  ReadStreamByBuf(in, limit, [&crc](const char* buf, size_t len) {
    crc = crc32(crc, reinterpret_cast<const Bytef*>(buf), len);
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

  auto end_pos = sout.tellg();
  sout.seekg(0, std::ios::beg);
  uint32_t len = static_cast<uint32_t>(end_pos - sout.tellg());
  uint32_t crc = Crc32Stream(sout);
  sout.seekg(0, std::ios::beg);

  Header hdr(static_cast<uint32_t>(records_.size()), crc, ct, len);
  hdr.Write(os);
  PipeStream(sout, os);
  return true;
}

void Chunk::Parse(std::istream& sin) {
  Header hdr;
  hdr.Parse(sin);
  auto beg_pos = sin.tellg();
  auto crc = Crc32Stream(sin, hdr.CompressSize());
  PADDLE_ENFORCE_EQ(hdr.Checksum(), crc);

  Clear();

  sin.seekg(beg_pos, std::ios::beg);
  std::unique_ptr<std::istream> compressed_stream;
  switch (hdr.CompressType()) {
    case Compressor::kNoCompress:
      break;
    case Compressor::kSnappy:
      compressed_stream.reset(new snappy::iSnappyStream(sin));
      break;
    default:
      PADDLE_THROW("Not implemented");
  }

  std::istream& stream = compressed_stream ? *compressed_stream : sin;

  for (uint32_t i = 0; i < hdr.NumRecords(); ++i) {
    uint32_t rec_len;
    stream.read(reinterpret_cast<char*>(&rec_len), sizeof(uint32_t));
    std::string buf;
    buf.resize(rec_len);
    stream.read(&buf[0], rec_len);
    Add(buf);
  }
}

}  // namespace recordio
}  // namespace paddle
