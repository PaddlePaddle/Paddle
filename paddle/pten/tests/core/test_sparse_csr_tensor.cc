/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gtest/gtest.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/sparse_csr_tensor.h"
#include "paddle/pten/tests/core/allocator.h"

namespace pten {
namespace tests {

TEST(sparse_csr_tensor, construct) {
  auto dense_dims = paddle::framework::make_ddim({3, 3});
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0};
  std::vector<int64_t> crows_data = {0, 1, 1, 3};
  std::vector<int64_t> cols_data = {1, 0, 2};

  auto alloc = std::make_shared<FancyAllocator>();
  auto crows_dims =
      paddle::framework::make_ddim({static_cast<int>(crows_data.size())});
  DenseTensorMeta crows_meta(DataType::INT64, crows_dims, DataLayout::NCHW);
  DenseTensor crows(alloc, crows_meta);
  memcpy(crows.mutable_data<int64_t>(),
         &crows_data[0],
         crows_data.size() * sizeof(int64_t));

  auto cols_dims =
      paddle::framework::make_ddim({static_cast<int>(cols_data.size())});
  DenseTensorMeta cols_meta(DataType::INT64, cols_dims, DataLayout::NCHW);
  DenseTensor cols(alloc, cols_meta);
  memcpy(cols.mutable_data<int64_t>(),
         &cols_data[0],
         cols_data.size() * sizeof(int64_t));

  auto values_dims =
      paddle::framework::make_ddim({static_cast<int>(non_zero_data.size())});
  DenseTensorMeta values_meta(DataType::FLOAT32, values_dims, DataLayout::NCHW);
  DenseTensor values(alloc, values_meta);
  memcpy(values.mutable_data<float>(),
         &non_zero_data[0],
         non_zero_data.size() * sizeof(float));

  SparseCsrTensor sparse(crows, cols, values, dense_dims);

  CHECK_EQ(sparse.nnz(), static_cast<int64_t>(non_zero_data.size()));
  CHECK_EQ(sparse.numel(), 9);
  CHECK(sparse.dims() == dense_dims);
  CHECK(sparse.dtype() == DataType::FLOAT32);
  CHECK(sparse.layout() == DataLayout::SPARSE_CSR);
  CHECK(sparse.place() == paddle::platform::CPUPlace());
  CHECK(sparse.initialized() == true);
}

}  // namespace tests
}  // namespace pten
