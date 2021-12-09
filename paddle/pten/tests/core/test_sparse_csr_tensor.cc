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
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {2.0, 0.0, 3.0}};
  auto dense_dims = paddle::framework::make_ddim({3, 3});
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0};
  std::vector<int64_t> crows_data = {0, 1, 1, 3};
  std::vector<int64_t> cols_data = {1, 0, 2};

  auto alloc = std::make_shared<FancyAllocator>();
  auto crows_dims =
      paddle::framework::make_ddim({static_cast<int>(crows_data.size())});
  DenseTensorMeta crows_meta(DataType::INT64, crows_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> crows_ptr(new DenseTensor(alloc, crows_meta));
  memcpy(crows_ptr->mutable_data<int64_t>(),
         &crows_data[0],
         crows_data.size() * sizeof(int64_t));

  auto cols_dims =
      paddle::framework::make_ddim({static_cast<int>(cols_data.size())});
  DenseTensorMeta cols_meta(DataType::INT64, cols_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> cols_ptr(new DenseTensor(alloc, cols_meta));
  memcpy(cols_ptr->mutable_data<int64_t>(),
         &cols_data[0],
         cols_data.size() * sizeof(int64_t));

  auto values_dims =
      paddle::framework::make_ddim({static_cast<int>(non_zero_data.size())});
  DenseTensorMeta values_meta(DataType::FLOAT32, values_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> values_ptr(new DenseTensor(alloc, values_meta));
  memcpy(
      values_ptr->mutable_data<float>(), &dense_data[0][0], 9 * sizeof(float));

  SparseCsrTensor sparse(std::move(crows_ptr),
                         std::move(cols_ptr),
                         std::move(values_ptr),
                         dense_dims);

  CHECK(sparse.nnz() == static_cast<int64_t>(non_zero_data.size()));
}

}  // namespace tests
}  // namespace pten
