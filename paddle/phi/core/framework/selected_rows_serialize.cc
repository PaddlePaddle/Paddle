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

#include "paddle/phi/core/framework/selected_rows_serialize.h"

namespace phi {

void SerializeToStream(std::ostream& os,
                       const phi::SelectedRows& selected_rows,
                       const phi::DeviceContext& dev_ctx) {
  {  // the 1st field, uint32_t version
    constexpr uint32_t version = 0;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  }
  {
    // the 2st field, rows information
    auto& rows = selected_rows.rows();
    uint64_t size = rows.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (uint64_t i = 0; i < size; ++i) {
      os.write(reinterpret_cast<const char*>(&rows[i]), sizeof(rows[i]));
    }
  }
  {
    // the 3st field, the height of SelectedRows
    int64_t height = selected_rows.height();
    os.write(reinterpret_cast<const char*>(&height), sizeof(height));
  }
  // the 4st field, Tensor data
  TensorToStream(os, selected_rows.value(), dev_ctx);
}

void SerializeToStream(std::ostream& os,
                       const phi::SelectedRows& selected_rows) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  const phi::DeviceContext* dev_ctx = nullptr;
  auto place = selected_rows.place();
  dev_ctx = pool.Get(place);
  SerializeToStream(os, selected_rows, *dev_ctx);
}

void DeserializeFromStream(std::istream& is, phi::SelectedRows* selected_rows) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  const phi::DeviceContext* dev_ctx = nullptr;
  dev_ctx = pool.Get(phi::CPUPlace());
  DeserializeFromStream(is, selected_rows, *dev_ctx);
}

void DeserializeFromStream(std::istream& is,
                           phi::SelectedRows* selected_rows,
                           const phi::DeviceContext& dev_ctx) {
  {
    // the 1st field, unit32_t version for SelectedRows
    uint32_t version = 0;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(version,
                      0U,
                      common::errors::InvalidArgument(
                          "Only version 0 SelectedRows is supported."));
  }
  {
    // the 2st field, rows information
    uint64_t size = 0;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    PADDLE_ENFORCE_EQ(
        is.good(),
        true,
        common::errors::Unavailable("Cannot read the number of rows."));
    auto& rows = *selected_rows->mutable_rows();
    rows.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
      is.read(reinterpret_cast<char*>(&rows[i]), sizeof(int64_t));
    }
  }
  {
    // the 3st field, the height of the SelectedRows
    int64_t height = 0;
    is.read(reinterpret_cast<char*>(&height), sizeof(int64_t));
    selected_rows->set_height(height);
  }
  // the 4st field, tensor which contains the data
  TensorFromStream(is, selected_rows->mutable_value(), dev_ctx);
}

}  // namespace phi
