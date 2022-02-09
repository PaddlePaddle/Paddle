/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See
the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <memory>

#include "paddle/pten/api/include/api.h"

#include "paddle/pten/api/include/sparse_api.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/sparse_coo_tensor.h"

TEST(API, to_sparse_coo) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            pten::framework::make_ddim({3, 3}),
                            pten::DataLayout::NCHW));

  pten::CPUPlace cpu;
  const int64_t sparse_dim = 2;
  auto* dense_x_data = dense_x->mutable_data<float>(cpu);
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const int64_t non_zero_num = 4;

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  pten::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();

  // 1. test dense_to_sparse_coo
  paddle::experimental::Tensor x(dense_x);
  auto out = paddle::experimental::sparse::to_sparse_coo(
      x, pten::Backend::CPU, sparse_dim);
  auto coo = std::dynamic_pointer_cast<pten::SparseCooTensor>(out.impl());
  ASSERT_EQ(coo->nnz(), non_zero_num);
  int cmp_indices = memcmp(coo->non_zero_indices().data<int64_t>(),
                           indices_data.data(),
                           indices_data.size() * sizeof(int64_t));
  ASSERT_EQ(cmp_indices, 0);
  int cmp_elements = memcmp(coo->non_zero_elements().data<float>(),
                            non_zero_data.data(),
                            non_zero_data.size() * sizeof(float));
  ASSERT_EQ(cmp_elements, 0);

  // 1. test sparse_csr_to_coo
  auto dense_dims = pten::framework::make_ddim({3, 3});
  pten::DenseTensorMeta crows_meta(
      pten::DataType::INT64, {dense_dims[0] + 1}, pten::DataLayout::NCHW);
  pten::DenseTensorMeta cols_meta(
      pten::DataType::INT64, {non_zero_num}, pten::DataLayout::NCHW);
  pten::DenseTensorMeta values_meta(
      pten::DataType::FLOAT32, {non_zero_num}, pten::DataLayout::NCHW);

  pten::CPUPlace place;
  pten::DenseTensor crows(alloc.get(), crows_meta);
  pten::DenseTensor cols(alloc.get(), cols_meta);
  pten::DenseTensor values(alloc.get(), values_meta);
  memcpy(crows.mutable_data<int64_t>(place),
         crows_data.data(),
         crows_data.size() * sizeof(int64_t));
  memcpy(cols.mutable_data<int64_t>(place),
         cols_data.data(),
         cols_data.size() * sizeof(int64_t));
  memcpy(values.mutable_data<float>(place),
         non_zero_data.data(),
         non_zero_data.size() * sizeof(float));
  auto csr =
      std::make_shared<pten::SparseCsrTensor>(crows, cols, values, dense_dims);
  paddle::experimental::Tensor csr_x(csr);
  auto out2 = paddle::experimental::sparse::to_sparse_coo(
      csr_x, pten::Backend::CPU, sparse_dim);

  auto coo2 = std::dynamic_pointer_cast<pten::SparseCooTensor>(out.impl());
  ASSERT_EQ(coo2->nnz(), non_zero_num);
  int cmp_indices2 = memcmp(coo2->non_zero_indices().data<int64_t>(),
                            indices_data.data(),
                            indices_data.size() * sizeof(int64_t));
  ASSERT_EQ(cmp_indices2, 0);
  int cmp_elements2 = memcmp(coo2->non_zero_elements().data<float>(),
                             non_zero_data.data(),
                             non_zero_data.size() * sizeof(float));
  ASSERT_EQ(cmp_elements2, 0);
}

TEST(API, to_sparse_csr) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            pten::framework::make_ddim({3, 3}),
                            pten::DataLayout::NCHW));

  pten::CPUPlace cpu;
  const int64_t sparse_dim = 2;
  auto* dense_x_data = dense_x->mutable_data<float>(cpu);
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const int64_t non_zero_num = 4;

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  pten::CPUContext dev_ctx_cpu;

  // 1. test dense_to_sparse_csr
  paddle::experimental::Tensor x(dense_x);
  auto out = paddle::experimental::sparse::to_sparse_csr(x, pten::Backend::CPU);
  auto csr = std::dynamic_pointer_cast<pten::SparseCsrTensor>(out.impl());
  auto check = [&](const pten::SparseCsrTensor& csr) {
    ASSERT_EQ(csr.non_zero_cols().numel(), non_zero_num);
    int cmp_crows = memcmp(csr.non_zero_crows().data<int64_t>(),
                           crows_data.data(),
                           crows_data.size() * sizeof(int64_t));
    ASSERT_EQ(cmp_crows, 0);
    int cmp_cols = memcmp(csr.non_zero_cols().data<int64_t>(),
                          cols_data.data(),
                          cols_data.size() * sizeof(int64_t));
    ASSERT_EQ(cmp_cols, 0);
    int cmp_elements = memcmp(csr.non_zero_elements().data<float>(),
                              non_zero_data.data(),
                              non_zero_data.size() * sizeof(float));
    ASSERT_EQ(cmp_elements, 0);
  };
  check(*csr);

  // 1. test sparse_coo_to_csr
  auto dense_dims = pten::framework::make_ddim({3, 3});
  pten::DenseTensorMeta indices_meta(pten::DataType::INT64,
                                     {sparse_dim, non_zero_num},
                                     pten::DataLayout::NCHW);
  pten::DenseTensorMeta values_meta(
      pten::DataType::FLOAT32, {non_zero_num}, pten::DataLayout::NCHW);

  pten::CPUPlace place;
  pten::DenseTensor indices(alloc.get(), indices_meta);
  pten::DenseTensor values(alloc.get(), values_meta);
  memcpy(indices.mutable_data<int64_t>(place),
         indices_data.data(),
         indices_data.size() * sizeof(int64_t));
  memcpy(values.mutable_data<float>(place),
         non_zero_data.data(),
         non_zero_data.size() * sizeof(float));
  auto coo =
      std::make_shared<pten::SparseCooTensor>(indices, values, dense_dims);
  paddle::experimental::Tensor coo_x(coo);
  auto out2 =
      paddle::experimental::sparse::to_sparse_csr(coo_x, pten::Backend::CPU);

  auto csr2 = std::dynamic_pointer_cast<pten::SparseCsrTensor>(out.impl());
  check(*csr2);
}

TEST(API, to_dense) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  pten::CPUPlace cpu;
  const int64_t sparse_dim = 2;
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const int64_t non_zero_num = 4;
  auto dense_dims = pten::framework::make_ddim({3, 3});

  pten::CPUContext dev_ctx_cpu;

  // 1. test sparse_coo_to_dense
  pten::DenseTensorMeta indices_meta(pten::DataType::INT64,
                                     {sparse_dim, non_zero_num},
                                     pten::DataLayout::NCHW);
  pten::DenseTensorMeta values_meta(
      pten::DataType::FLOAT32, {non_zero_num}, pten::DataLayout::NCHW);

  pten::CPUPlace place;
  pten::DenseTensor indices(alloc.get(), indices_meta);
  pten::DenseTensor values(alloc.get(), values_meta);
  memcpy(indices.mutable_data<int64_t>(place),
         indices_data.data(),
         indices_data.size() * sizeof(int64_t));
  memcpy(values.mutable_data<float>(place),
         non_zero_data.data(),
         non_zero_data.size() * sizeof(float));
  auto coo =
      std::make_shared<pten::SparseCooTensor>(indices, values, dense_dims);

  paddle::experimental::Tensor coo_x(coo);
  auto out = paddle::experimental::sparse::to_dense(coo_x, pten::Backend::CPU);
  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  int cmp1 =
      memcmp(dense_out->data<float>(), &dense_data[0][0], 9 * sizeof(float));
  ASSERT_EQ(cmp1, 0);

  // 1. test sparse_csr_to_dense
  pten::DenseTensorMeta crows_meta(
      pten::DataType::INT64, {dense_dims[0] + 1}, pten::DataLayout::NCHW);
  pten::DenseTensorMeta cols_meta(
      pten::DataType::INT64, {non_zero_num}, pten::DataLayout::NCHW);
  pten::DenseTensor crows(alloc.get(), crows_meta);
  pten::DenseTensor cols(alloc.get(), cols_meta);
  memcpy(crows.mutable_data<int64_t>(place),
         crows_data.data(),
         crows_data.size() * sizeof(int64_t));
  memcpy(cols.mutable_data<int64_t>(place),
         cols_data.data(),
         cols_data.size() * sizeof(int64_t));
  memcpy(values.mutable_data<float>(place),
         non_zero_data.data(),
         non_zero_data.size() * sizeof(float));
  auto csr =
      std::make_shared<pten::SparseCsrTensor>(crows, cols, values, dense_dims);
  paddle::experimental::Tensor csr_x(csr);
  auto out2 = paddle::experimental::sparse::to_dense(csr_x, pten::Backend::CPU);

  auto dense_out2 = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  int cmp2 =
      memcmp(dense_out2->data<float>(), &dense_data[0][0], 9 * sizeof(float));
  ASSERT_EQ(cmp2, 0);
}
