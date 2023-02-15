/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/serialization.h"

#include "paddle/phi/core/enforce.h"

// Note: The TensorToStream depends on framework.proto,
// it is difficult to move into phi
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/version.h"

namespace phi {

void SerializeToStream(std::ostream &os,
                       const DenseTensor &tensor,
                       const DeviceContext &dev_ctx) {
  {  // the 1st field, uint32_t version for DenseTensor
    os.write(
        reinterpret_cast<const char *>(&paddle::framework::kCurTensorVersion),
        sizeof(paddle::framework::kCurTensorVersion));
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
      size = each.size() * sizeof(phi::LoD::value_type::value_type);
      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      os.write(reinterpret_cast<const char *>(each.data()),
               static_cast<std::streamsize>(size));
    }
  }
  // the 3st field, Tensor
  paddle::framework::TensorToStream(
      os, static_cast<DenseTensor>(tensor), dev_ctx);
}

void DeserializeFromStream(std::istream &is,
                           DenseTensor *tensor,
                           const DeviceContext &dev_ctx,
                           const size_t &seek,
                           const std::vector<int64_t> &shape) {
  {
    // the 1st field, unit32_t version for DenseTensor
    uint32_t version;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(paddle::framework::IsTensorVersionSupported(version),
                      true,
                      phi::errors::InvalidArgument(
                          "Tensor version %u is not supported.", version));
    PADDLE_ENFORCE_EQ(
        version,
        0U,
        phi::errors::InvalidArgument(
            "Deserialize to tensor failed, maybe the loaded file is "
            "not a paddle model(expected file format: 0, but %u found).",
            version));
  }
  {
    // the 2st field, LoD information
    uint64_t lod_level;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
  }
  // the 3st filed, Tensor
  paddle::framework::TensorFromStream(
      is, static_cast<DenseTensor *>(tensor), dev_ctx, seek, shape);
}

void DeserializeFromStream(std::istream &is,
                           DenseTensor *tensor,
                           const DeviceContext &dev_ctx) {
  {
    // the 1st field, unit32_t version for DenseTensor
    uint32_t version;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(paddle::framework::IsTensorVersionSupported(version),
                      true,
                      phi::errors::InvalidArgument(
                          "Tensor version %u is not supported.", version));
    PADDLE_ENFORCE_EQ(
        version,
        0U,
        phi::errors::InvalidArgument(
            "Deserialize to tensor failed, maybe the loaded file is "
            "not a paddle model(expected file format: 0, but %u found).",
            version));
  }
  {
    // the 2st field, LoD information
    uint64_t lod_level;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
    for (uint64_t i = 0; i < lod_level; ++i) {
      uint64_t size;
      is.read(reinterpret_cast<char *>(&size), sizeof(size));
      std::vector<size_t> tmp(size / sizeof(size_t));
      is.read(reinterpret_cast<char *>(tmp.data()),
              static_cast<std::streamsize>(size));
      lod[i] = tmp;
    }
  }
  // the 3st filed, Tensor
  paddle::framework::TensorFromStream(
      is, static_cast<DenseTensor *>(tensor), dev_ctx);
}

void SerializeToStream(std::ostream &os,
                       const SelectedRows &selected_rows,
                       const DeviceContext &dev_ctx) {
  {  // the 1st field, uint32_t version
    constexpr uint32_t version = 0;
    os.write(reinterpret_cast<const char *>(&version), sizeof(version));
  }
  {
    // the 2st field, rows information
    auto &rows = selected_rows.rows();
    uint64_t size = rows.size();
    os.write(reinterpret_cast<const char *>(&size), sizeof(size));
    for (uint64_t i = 0; i < size; ++i) {
      os.write(reinterpret_cast<const char *>(&rows[i]), sizeof(rows[i]));
    }
  }
  {
    // the 3st field, the height of SelectedRows
    int64_t height = selected_rows.height();
    os.write(reinterpret_cast<const char *>(&height), sizeof(height));
  }
  // the 4st field, Tensor data
  paddle::framework::TensorToStream(os, selected_rows.value(), dev_ctx);
}

void DeserializeFromStream(std::istream &is,
                           SelectedRows *selected_rows,
                           const DeviceContext &dev_ctx) {
  {
    // the 1st field, unit32_t version for SelectedRows
    uint32_t version;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(version,
                      0U,
                      phi::errors::InvalidArgument(
                          "Only version 0 SelectedRows is supported."));
  }
  {
    // the 2st field, rows information
    uint64_t size = 0;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    PADDLE_ENFORCE_EQ(
        is.good(),
        true,
        phi::errors::Unavailable("Cannot read the number of rows."));
    auto &rows = *selected_rows->mutable_rows();
    rows.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
      is.read(reinterpret_cast<char *>(&rows[i]), sizeof(int64_t));
    }
  }
  {
    // the 3st field, the height of the SelectedRows
    int64_t height;
    is.read(reinterpret_cast<char *>(&height), sizeof(int64_t));
    selected_rows->set_height(height);
  }
  // the 4st field, tensor which contains the data
  paddle::framework::TensorFromStream(
      is, selected_rows->mutable_value(), dev_ctx);
}

}  // namespace phi
