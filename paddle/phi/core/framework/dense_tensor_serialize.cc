// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/framework/dense_tensor_serialize.h"
#include <cstdint>
#include <limits>
#include "paddle/phi/core/framework/convert_utils.h"

namespace phi {

namespace {
// Returns the number of bytes remaining to be read from a seekable stream, or
// -1 if the stream does not support seeking (e.g. a pure input pipe). Used to
// bound attacker-controlled length fields before allocating, so a malformed
// serialized tensor cannot force a huge allocation (OOM).
int64_t GetRemainingStreamBytes(std::istream &is) {
  const std::istream::pos_type cur = is.tellg();
  if (cur == std::istream::pos_type(-1)) {
    return -1;
  }
  is.seekg(0, std::ios::end);
  const std::istream::pos_type end = is.tellg();
  is.seekg(cur);
  if (end == std::istream::pos_type(-1)) {
    return -1;
  }
  return static_cast<int64_t>(end - cur);
}
}  // namespace

void SerializeToStream(std::ostream &os,
                       const DenseTensor &tensor,
                       const DeviceContext &dev_ctx) {
  constexpr uint32_t kCurTensorVersion = 0;
  {  // the 1st field, uint32_t version for DenseTensor
    os.write(reinterpret_cast<const char *>(&kCurTensorVersion),
             sizeof(kCurTensorVersion));
  }
  {
    // the 2nd field, LoD information
    // uint64_t lod_level
    // uint64_t lod_level_1 size in byte.
    // int*     lod_level_1 data
    // ...
    auto lod = tensor.lod();
    uint64_t size = lod.size();
    os.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (auto &each : lod) {
      size = each.size() * sizeof(LegacyLoD::value_type::value_type);
      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      os.write(reinterpret_cast<const char *>(each.data()),
               static_cast<std::streamsize>(size));
    }
  }
  // the 3rd field, Tensor
  TensorToStream(os, static_cast<DenseTensor>(tensor), dev_ctx);
}

void SerializeToStream(std::ostream &os, const DenseTensor &tensor) {
  DeviceContextPool &pool = DeviceContextPool::Instance();
  const DeviceContext *dev_ctx = nullptr;
  auto place = tensor.place();
  dev_ctx = pool.Get(place);
  SerializeToStream(os, tensor, *dev_ctx);
}

void DeserializeFromStream(std::istream &os, DenseTensor *tensor) {
  DeviceContextPool &pool = DeviceContextPool::Instance();
  const DeviceContext *dev_ctx = nullptr;
  dev_ctx = pool.Get(CPUPlace());
  DeserializeFromStream(os, tensor, *dev_ctx);
}

void DeserializeFromStream(std::istream &is,
                           DenseTensor *tensor,
                           const DeviceContext &dev_ctx,
                           const size_t &seek,
                           const std::vector<int64_t> &shape) {
  {
    // the 1st field, unit32_t version for DenseTensor
    uint32_t version = 0;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));

    PADDLE_ENFORCE_EQ(
        version,
        0U,
        common::errors::InvalidArgument(
            "Deserialize to tensor failed, maybe the loaded file is "
            "not a paddle model(expected file format: 0, but %u found).",
            version));
  }
  {
    // the 2nd field, LoD information
    uint64_t lod_level = 0;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    // `lod_level` is attacker-controlled. Ensure the header was fully read and
    // bound it by the number of LoD levels the remaining stream could hold
    // (each level needs at least one uint64_t) before `resize` allocates, so a
    // crafted file cannot force a huge allocation (OOM).
    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(is.gcount()),
        sizeof(lod_level),
        common::errors::InvalidArgument(
            "Deserialize LoD information failed, expected to read %zu bytes "
            "for the LoD level count but only %lld bytes were available. The "
            "input stream may be truncated or corrupted.",
            sizeof(lod_level),
            static_cast<int64_t>(is.gcount())));
    const int64_t remaining_bytes = GetRemainingStreamBytes(is);
    if (remaining_bytes >= 0) {
      PADDLE_ENFORCE_LE(
          lod_level,
          static_cast<uint64_t>(remaining_bytes) / sizeof(uint64_t),
          common::errors::InvalidArgument(
              "Deserialize LoD information failed, the declared LoD level "
              "count (%llu) exceeds what the remaining %lld bytes of the input "
              "stream can hold. The input stream may be corrupted or "
              "malicious.",
              lod_level,
              static_cast<int64_t>(remaining_bytes)));
    }
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
  }
  // the 3rd field, Tensor
  TensorFromStream(
      is, static_cast<DenseTensor *>(tensor), dev_ctx, seek, shape);
}

void DeserializeFromStream(std::istream &is,
                           DenseTensor *tensor,
                           const DeviceContext &dev_ctx) {
  {
    // the 1st field, unit32_t version for DenseTensor
    uint32_t version = 0;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));

    PADDLE_ENFORCE_EQ(
        version,
        0U,
        common::errors::InvalidArgument(
            "Deserialize to tensor failed, maybe the loaded file is "
            "not a paddle model(expected file format: 0, but %u found).",
            version));
  }
  {
    // the 2nd field, LoD information
    uint64_t lod_level = 0;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    // `lod_level` is attacker-controlled. Ensure the header was fully read and
    // bound it by the number of LoD levels the remaining stream could hold
    // (each level needs at least one uint64_t size field) before `resize`
    // allocates, so a crafted file cannot force a huge allocation (OOM).
    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(is.gcount()),
        sizeof(lod_level),
        common::errors::InvalidArgument(
            "Deserialize LoD information failed, expected to read %zu bytes "
            "for the LoD level count but only %lld bytes were available. The "
            "input stream may be truncated or corrupted.",
            sizeof(lod_level),
            static_cast<int64_t>(is.gcount())));
    const int64_t remaining_bytes = GetRemainingStreamBytes(is);
    if (remaining_bytes >= 0) {
      PADDLE_ENFORCE_LE(
          lod_level,
          static_cast<uint64_t>(remaining_bytes) / sizeof(uint64_t),
          common::errors::InvalidArgument(
              "Deserialize LoD information failed, the declared LoD level "
              "count (%llu) exceeds what the remaining %lld bytes of the input "
              "stream can hold. The input stream may be corrupted or "
              "malicious.",
              lod_level,
              static_cast<int64_t>(remaining_bytes)));
    }
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
    for (uint64_t i = 0; i < lod_level; ++i) {
      uint64_t size = 0;
      is.read(reinterpret_cast<char *>(&size), sizeof(size));
      PADDLE_ENFORCE_EQ(
          static_cast<uint64_t>(is.gcount()),
          sizeof(size),
          common::errors::InvalidArgument(
              "Deserialize LoD information failed, expected to read %zu bytes "
              "for the byte size of LoD level %llu but only %lld bytes were "
              "available. The input stream may be truncated or corrupted.",
              sizeof(size),
              i,
              static_cast<int64_t>(is.gcount())));
      // The destination buffer holds `size / sizeof(size_t)` elements, i.e.
      // exactly `size` bytes only when `size` is a multiple of `sizeof(size_t)`.
      // Otherwise the read below would write past the allocation (CWE-787 /
      // heap buffer overflow), so reject such sizes.
      PADDLE_ENFORCE_EQ(
          size % sizeof(size_t),
          0,
          common::errors::InvalidArgument(
              "Deserialize LoD information failed, the byte size of LoD level "
              "%llu is %llu, which is not a multiple of sizeof(size_t) (%zu). "
              "The input stream may be corrupted or malicious.",
              i,
              size,
              sizeof(size_t)));
      // Bound `size` before allocating so a crafted file cannot force a huge
      // allocation (OOM), and guard the `std::streamsize` cast used by `read`.
      PADDLE_ENFORCE_LE(
          size,
          static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()),
          common::errors::InvalidArgument(
              "Deserialize LoD information failed, the byte size of LoD level "
              "%llu (%llu) exceeds the maximum readable stream size. The input "
              "stream may be corrupted or malicious.",
              i,
              size));
      const int64_t level_remaining_bytes = GetRemainingStreamBytes(is);
      if (level_remaining_bytes >= 0) {
        PADDLE_ENFORCE_LE(
            size,
            static_cast<uint64_t>(level_remaining_bytes),
            common::errors::InvalidArgument(
                "Deserialize LoD information failed, the byte size of LoD "
                "level %llu (%llu) exceeds the remaining %lld bytes of the "
                "input stream. The input stream may be corrupted or malicious.",
                i,
                size,
                static_cast<int64_t>(level_remaining_bytes)));
      }
      std::vector<size_t> tmp(size / sizeof(size_t));
      is.read(reinterpret_cast<char *>(tmp.data()),
              static_cast<std::streamsize>(size));
      PADDLE_ENFORCE_EQ(
          static_cast<uint64_t>(is.gcount()),
          size,
          common::errors::InvalidArgument(
              "Deserialize LoD information failed, expected to read %llu bytes "
              "for LoD level %llu but only %lld bytes were available. The "
              "input stream may be truncated or corrupted.",
              size,
              i,
              static_cast<int64_t>(is.gcount())));
      lod[i] = tmp;
    }
  }
  // the 3rd field, Tensor
  TensorFromStream(is, static_cast<DenseTensor *>(tensor), dev_ctx);
}

}  // namespace phi
