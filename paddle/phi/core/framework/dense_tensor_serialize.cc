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
                       const phi::DenseTensor &tensor,
                       const phi::DeviceContext &dev_ctx) {
  constexpr uint32_t kCurTensorVersion = 0;
  {  // the 1st field, uint32_t version for DenseTensor
    os.write(reinterpret_cast<const char *>(&kCurTensorVersion),
             sizeof(kCurTensorVersion));
  }
  {
    // the 2st field, LoD information
    // uint64_t lod_level
    // uint64_t lod_level_1 size in byte.
    // int*     lod_level_1 data
    // ...
    auto lod = tensor.lod();
    uint64_t size = lod.size();
    os.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (auto &each : lod) {
      size = each.size() * sizeof(phi::LegacyLoD::value_type::value_type);
      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      os.write(reinterpret_cast<const char *>(each.data()),
               static_cast<std::streamsize>(size));
    }
  }
  // the 3st field, Tensor
  TensorToStream(os, static_cast<phi::DenseTensor>(tensor), dev_ctx);
}

void SerializeToStream(std::ostream &os, const phi::DenseTensor &tensor) {
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  const phi::DeviceContext *dev_ctx = nullptr;
  auto place = tensor.place();
  dev_ctx = pool.Get(place);
  SerializeToStream(os, tensor, *dev_ctx);
}

void DeserializeFromStream(std::istream &os, phi::DenseTensor *tensor) {
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  const phi::DeviceContext *dev_ctx = nullptr;
  dev_ctx = pool.Get(phi::CPUPlace());
  DeserializeFromStream(os, tensor, *dev_ctx);
}

void DeserializeFromStream(std::istream &is,
                           phi::DenseTensor *tensor,
                           const phi::DeviceContext &dev_ctx,
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
    // the 2st field, LoD information
    uint64_t lod_level = 0;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
  }
  // the 3st filed, Tensor
  TensorFromStream(
      is, static_cast<phi::DenseTensor *>(tensor), dev_ctx, seek, shape);
}

void DeserializeFromStream(std::istream &is,
                           phi::DenseTensor *tensor,
                           const phi::DeviceContext &dev_ctx) {
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
    // the 2st field, LoD information
    uint64_t lod_level = 0;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
    for (uint64_t i = 0; i < lod_level; ++i) {
      uint64_t size = 0;
      is.read(reinterpret_cast<char *>(&size), sizeof(size));
      std::vector<size_t> tmp(size / sizeof(size_t));
      is.read(reinterpret_cast<char *>(tmp.data()),
              static_cast<std::streamsize>(size));
      lod[i] = tmp;
    }
  }
  // the 3st filed, Tensor
  TensorFromStream(is, static_cast<phi::DenseTensor *>(tensor), dev_ctx);
}

}  // namespace phi
