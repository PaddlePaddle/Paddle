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
#include "paddle/phi/core/framework/convert_utils.h"

namespace phi {

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
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
    for (uint64_t i = 0; i < lod_level; ++i) {
      uint64_t size = 0;
      is.read(reinterpret_cast<char *>(&size), sizeof(size));
      // `size` is an attacker-controlled byte count read from the stream. The
      // destination buffer holds `size / sizeof(size_t)` elements, i.e.
      // `(size / sizeof(size_t)) * sizeof(size_t)` bytes, but the read below
      // consumes the full `size` bytes. If `size` is not a multiple of
      // `sizeof(size_t)`, the read writes past the allocation (CWE-787 / heap
      // buffer overflow). Reject such sizes before allocating/reading.
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
