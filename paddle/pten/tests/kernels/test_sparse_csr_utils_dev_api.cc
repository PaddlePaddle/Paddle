/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF NCHW KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <memory>

#include "paddle/pten/include/sparse_csr_tensor_utils.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/gpu/utils.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(DEV_API, to_sparse_csr_cpu) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(alloc,
                            pten::DenseTensorMeta(pten::DataType::FLOAT32,
                                                  framework::make_ddim({3, 3}),
                                                  pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x.mutable_data<float>();
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

  // 2. test API
  auto sparse_out = pten::ToSparseCsr<float>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)), dense_x);
  int64_t non_zero_num = sparse_out.nnz();
  ASSERT_EQ(static_cast<uint64_t>(non_zero_num), non_zero_data.size());

  const auto& crows = sparse_out.non_zero_crows();
  const auto& cols = sparse_out.non_zero_cols();
  const auto& non_zero_elements = sparse_out.non_zero_elements();
  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(non_zero_elements.data<float>()[i], non_zero_data[i]);
    ASSERT_EQ(cols.data<int64_t>()[i], cols_data[i]);
  }
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(crows.data<int64_t>()[i], crows_data[i]);
  }
}

TEST(DEV_API, to_sparse_csr_cuda) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());

  pten::DenseTensor dense_x(alloc,
                            pten::DenseTensorMeta(pten::DataType::FLOAT32,
                                                  framework::make_ddim({3, 3}),
                                                  pten::DataLayout::NCHW));

  pten::DenseTensor d_dense_x(
      cuda_alloc,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 3}),
                            pten::DataLayout::NCHW));

  auto* dense_x_data = dense_x.mutable_data<float>();
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  // auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());
  auto* dev_ctx_cuda = pool.Get(paddle::platform::CUDAPlace());
  auto* dev_ctx =
      static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda);

  pten::Copy(*dev_ctx, dense_x, true, &d_dense_x);

  // 2. test API
  auto sparse_out = pten::ToSparseCsr<float>(
      *(static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda)),
      d_dense_x);

  int64_t non_zero_num = sparse_out.nnz();
  ASSERT_EQ(static_cast<uint64_t>(non_zero_num), non_zero_data.size());

  const auto& d_crows = sparse_out.non_zero_crows();
  const auto& d_cols = sparse_out.non_zero_cols();
  const auto& d_non_zero_elements = sparse_out.non_zero_elements();
  pten::DenseTensor crows(
      alloc,
      pten::DenseTensorMeta(
          pten::DataType::INT64, d_crows.dims(), d_crows.layout()));
  pten::DenseTensor cols(
      alloc,
      pten::DenseTensorMeta(
          pten::DataType::INT64, d_cols.dims(), d_cols.layout()));

  pten::DenseTensor non_zero_elements(
      alloc,
      pten::DenseTensorMeta(d_non_zero_elements.dtype(),
                            d_non_zero_elements.dims(),
                            d_non_zero_elements.layout()));
  pten::Copy(*dev_ctx, d_crows, true, &crows);
  pten::Copy(*dev_ctx, d_cols, true, &cols);
  pten::Copy(*dev_ctx, d_non_zero_elements, true, &non_zero_elements);

  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(non_zero_elements.data<float>()[i], non_zero_data[i]);
    ASSERT_EQ(cols.data<int64_t>()[i], cols_data[i]);
  }
  for (uint64_t i = 0; i < crows_data.size(); i++) {
    ASSERT_EQ(crows.data<int64_t>()[i], crows_data[i]);
  }
}

TEST(DEV_API, to_dense) {
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor h_crows(
      alloc,
      pten::DenseTensorMeta(
          pten::DataType::INT64,
          framework::make_ddim({static_cast<int64_t>(crows_data.size())}),
          pten::DataLayout::NCHW));
  pten::DenseTensor h_cols(
      alloc,
      pten::DenseTensorMeta(
          pten::DataType::INT64,
          framework::make_ddim({static_cast<int64_t>(cols_data.size())}),
          pten::DataLayout::NCHW));
  pten::DenseTensor h_values(
      alloc,
      pten::DenseTensorMeta(
          pten::DataType::FLOAT32,
          framework::make_ddim({static_cast<int64_t>(non_zero_data.size())}),
          pten::DataLayout::NCHW));

  std::copy(
      crows_data.begin(), crows_data.end(), h_crows.mutable_data<int64_t>());
  std::copy(cols_data.begin(), cols_data.end(), h_cols.mutable_data<int64_t>());
  std::copy(non_zero_data.begin(),
            non_zero_data.end(),
            h_values.mutable_data<float>());

  framework::DDim dense_dim = framework::make_ddim({3, 3});
  pten::SparseCsrTensor sparse_tensor(h_crows, h_cols, h_values, dense_dim);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());
  auto dense_out = pten::SparseCsrToDense<float>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)),
      sparse_tensor);

  for (int i = 0; i < dense_dim[0]; i++) {
    for (int j = 0; j < dense_dim[1]; j++) {
      ASSERT_EQ(dense_out.data<float>()[i * dense_dim[1] + j],
                dense_data[i][j]);
    }
  }
}

TEST(DEV_API, to_dense_cuda) {
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};

  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());

  pten::DenseTensor h_crows(
      alloc,
      pten::DenseTensorMeta(
          pten::DataType::INT64,
          framework::make_ddim({static_cast<int64_t>(crows_data.size())}),
          pten::DataLayout::NCHW));
  pten::DenseTensor h_cols(
      alloc,
      pten::DenseTensorMeta(
          pten::DataType::INT64,
          framework::make_ddim({static_cast<int64_t>(cols_data.size())}),
          pten::DataLayout::NCHW));
  pten::DenseTensor h_values(
      alloc,
      pten::DenseTensorMeta(
          pten::DataType::FLOAT32,
          framework::make_ddim({static_cast<int64_t>(non_zero_data.size())}),
          pten::DataLayout::NCHW));

  std::copy(
      crows_data.begin(), crows_data.end(), h_crows.mutable_data<int64_t>());
  std::copy(cols_data.begin(), cols_data.end(), h_cols.mutable_data<int64_t>());
  std::copy(non_zero_data.begin(),
            non_zero_data.end(),
            h_values.mutable_data<float>());

  pten::DenseTensor d_crows(
      cuda_alloc,
      pten::DenseTensorMeta(h_crows.dtype(), h_crows.dims(), h_crows.layout()));
  pten::DenseTensor d_cols(
      cuda_alloc,
      pten::DenseTensorMeta(h_cols.dtype(), h_cols.dims(), h_cols.layout()));
  pten::DenseTensor d_values(
      cuda_alloc,
      pten::DenseTensorMeta(
          h_values.dtype(), h_values.dims(), h_values.layout()));
  framework::DDim dense_dim = framework::make_ddim({3, 3});

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx_cuda = pool.Get(paddle::platform::CUDAPlace());
  auto* dev_ctx =
      static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda);
  pten::Copy(*dev_ctx, h_crows, true, &d_crows);
  pten::Copy(*dev_ctx, h_cols, true, &d_cols);
  pten::Copy(*dev_ctx, h_values, true, &d_values);
  pten::SparseCsrTensor sparse_tensor(d_crows, d_cols, d_values, dense_dim);
  auto dense_out = pten::SparseCsrToDense<float>(
      *(static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda)),
      sparse_tensor);
  pten::DenseTensor h_dense(alloc, dense_out.meta());
  pten::Copy(*dev_ctx, dense_out, true, &h_dense);

  for (int i = 0; i < dense_dim[0]; i++) {
    for (int j = 0; j < dense_dim[1]; j++) {
      ASSERT_EQ(h_dense.data<float>()[i * dense_dim[1] + j], dense_data[i][j]);
    }
  }
}

TEST(DEV_API, sparse_coo_to_csr_cpu) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(alloc,
                            pten::DenseTensorMeta(pten::DataType::FLOAT32,
                                                  framework::make_ddim({3, 3}),
                                                  pten::DataLayout::NCHW));
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> non_zero_indices = {0, 1, 1, 2, 1, 0, 2, 0};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const int64_t non_zero_num = 4;

  framework::DDim dense_dims = framework::make_ddim({3, 3});
  pten::DenseTensorMeta indices_meta(pten::DataType::INT64,
                                     framework::make_ddim({2, non_zero_num}),
                                     pten::DataLayout::NCHW);
  pten::DenseTensorMeta values_meta(pten::DataType::FLOAT32,
                                    framework::make_ddim({non_zero_num}),
                                    pten::DataLayout::NCHW);

  pten::DenseTensor indices(alloc, indices_meta);
  pten::DenseTensor values(alloc, values_meta);
  memcpy(indices.mutable_data<int64_t>(),
         non_zero_indices.data(),
         non_zero_indices.size() * sizeof(int64_t));
  memcpy(values.mutable_data<float>(),
         non_zero_data.data(),
         non_zero_data.size() * sizeof(float));
  pten::SparseCooTensor coo(indices, values, dense_dims);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

  // 2. test API
  auto sparse_out = pten::SparseCooToCsr<float>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)), coo);
  int64_t actual_non_zero_num = sparse_out.nnz();
  ASSERT_EQ(actual_non_zero_num, non_zero_num);

  const auto& crows = sparse_out.non_zero_crows();
  const auto& cols = sparse_out.non_zero_cols();
  const auto& non_zero_elements = sparse_out.non_zero_elements();
  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(non_zero_elements.data<float>()[i], non_zero_data[i]);
    ASSERT_EQ(cols.data<int64_t>()[i], cols_data[i]);
  }
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(crows.data<int64_t>()[i], crows_data[i]);
  }
}

TEST(DEV_API, sparse_csr_to_coo_cpu) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(alloc,
                            pten::DenseTensorMeta(pten::DataType::FLOAT32,
                                                  framework::make_ddim({3, 3}),
                                                  pten::DataLayout::NCHW));
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> non_zero_indices = {0, 1, 1, 2, 1, 0, 2, 0};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const int64_t non_zero_num = 4;

  framework::DDim dense_dims = framework::make_ddim({3, 3});
  pten::DenseTensorMeta crows_meta(pten::DataType::INT64,
                                   framework::make_ddim({dense_dims[0] + 1}),
                                   pten::DataLayout::NCHW);
  pten::DenseTensorMeta cols_meta(pten::DataType::INT64,
                                  framework::make_ddim({non_zero_num}),
                                  pten::DataLayout::NCHW);
  pten::DenseTensorMeta values_meta(pten::DataType::FLOAT32,
                                    framework::make_ddim({non_zero_num}),
                                    pten::DataLayout::NCHW);

  pten::DenseTensor crows(alloc, crows_meta);
  pten::DenseTensor cols(alloc, cols_meta);
  pten::DenseTensor values(alloc, values_meta);
  memcpy(crows.mutable_data<int64_t>(),
         crows_data.data(),
         crows_data.size() * sizeof(int64_t));
  memcpy(cols.mutable_data<int64_t>(),
         cols_data.data(),
         cols_data.size() * sizeof(int64_t));
  memcpy(values.mutable_data<float>(),
         non_zero_data.data(),
         non_zero_data.size() * sizeof(float));
  pten::SparseCsrTensor csr(crows, cols, values, dense_dims);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

  // 2. test API
  auto sparse_out = pten::SparseCsrToCoo<float>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)), csr);
  int64_t actual_non_zero_num = sparse_out.nnz();
  ASSERT_EQ(actual_non_zero_num, non_zero_num);

  const auto& indices = sparse_out.non_zero_indices();
  const auto& non_zero_elements = sparse_out.non_zero_elements();
  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(non_zero_elements.data<float>()[i], non_zero_data[i]);
    ASSERT_EQ(indices.data<int64_t>()[i], non_zero_indices[i]);
    ASSERT_EQ(indices.data<int64_t>()[i + non_zero_num],
              non_zero_indices[i + non_zero_num]);
  }
}
}  // namespace tests
}  // namespace pten
