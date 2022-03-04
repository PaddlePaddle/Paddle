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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/tests/core/allocator.h"

namespace phi {
namespace tests {

TEST(sparse_coo_tensor, construct) {
  phi::CPUPlace cpu;
  auto dense_dims = phi::make_ddim({3, 3});
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0};
  std::vector<int64_t> indices_data = {0, 1, 2, 0, 2, 1};
  auto fancy_allocator = std::unique_ptr<Allocator>(new FancyAllocator);
  auto* alloc = fancy_allocator.get();
  auto indices_dims =
      phi::make_ddim({2, static_cast<int>(non_zero_data.size())});
  DenseTensorMeta indices_meta(DataType::INT64, indices_dims, DataLayout::NCHW);
  DenseTensor indices(alloc, indices_meta);
  memcpy(indices.mutable_data<int64_t>(cpu),
         &indices_data[0],
         indices_data.size() * sizeof(int64_t));

  auto elements_dims = phi::make_ddim({static_cast<int>(non_zero_data.size())});
  DenseTensorMeta elements_meta(
      DataType::FLOAT32, elements_dims, DataLayout::NCHW);
  DenseTensor elements(alloc, elements_meta);

  memcpy(elements.mutable_data<float>(cpu),
         &non_zero_data[0],
         non_zero_data.size() * sizeof(float));

  SparseCooTensor sparse(indices, elements, dense_dims);

  CHECK(sparse.initialized() == true);
  CHECK_EQ(sparse.nnz(), static_cast<int64_t>(non_zero_data.size()));
  CHECK_EQ(sparse.numel(), 9);
  CHECK(sparse.dims() == dense_dims);
  CHECK(sparse.dtype() == DataType::FLOAT32);
  CHECK(sparse.layout() == DataLayout::SPARSE_COO);
  CHECK(sparse.place() == phi::CPUPlace());
}

TEST(sparse_coo_tensor, other_function) {
  auto fancy_allocator = std::unique_ptr<Allocator>(new FancyAllocator);
  auto* alloc = fancy_allocator.get();
  auto dense_dims = phi::make_ddim({4, 4});
  const int non_zero_num = 2;
  auto indices_dims = phi::make_ddim({2, non_zero_num});
  DenseTensorMeta indices_meta(DataType::INT64, indices_dims, DataLayout::NCHW);
  DenseTensor indices(alloc, indices_meta);

  auto elements_dims = phi::make_ddim({non_zero_num});
  DenseTensorMeta elements_meta(
      DataType::FLOAT32, elements_dims, DataLayout::NCHW);
  DenseTensor elements(alloc, elements_meta);

  SparseCooTensor coo(indices, elements, dense_dims);
  CHECK(coo.initialized());
  CHECK_EQ(coo.dims(), dense_dims);

  // Test Resize
  auto dense_dims_3d = phi::make_ddim({2, 4, 4});
  coo.Resize(dense_dims_3d, 1, 3);
  CHECK_EQ(coo.nnz(), 3);

  // Test shallow_copy
  SparseCooTensor coo2(coo);
  CHECK(coo.dims() == coo2.dims());

  // Test shallow_copy_assignment
  SparseCooTensor coo3 = coo2;
  CHECK(coo3.dims() == coo2.dims());
}

}  // namespace tests
}  // namespace phi
