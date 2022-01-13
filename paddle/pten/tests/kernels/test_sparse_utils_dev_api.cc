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

#include "paddle/pten/kernels/copy_kernel.h"
#include "paddle/pten/kernels/sparse_utils_kernel.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(DEV_API, to_sparse_coo_cuda) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());

  DenseTensor dense_x(
      alloc,
      DenseTensorMeta(
          DataType::FLOAT32, framework::make_ddim({3, 3}), DataLayout::NCHW));

  DenseTensor d_dense_x(
      cuda_alloc,
      DenseTensorMeta(
          DataType::FLOAT32, framework::make_ddim({3, 3}), DataLayout::NCHW));

  auto* dense_x_data = dense_x.mutable_data<float>();
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx_cuda = pool.Get(paddle::platform::CUDAPlace());
  auto* dev_ctx =
      static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda);

  pten::Copy(*dev_ctx, dense_x, true, &d_dense_x);

  // 2. test API
  auto sparse_out = DenseToSparseCoo<float>(
      *(static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda)),
      d_dense_x,
      2);

  int64_t non_zero_num = sparse_out.nnz();
  ASSERT_EQ(static_cast<uint64_t>(non_zero_num), non_zero_data.size());

  const auto& d_indices = sparse_out.non_zero_indices();
  const auto& d_non_zero_elements = sparse_out.non_zero_elements();
  DenseTensor indices(
      alloc,
      DenseTensorMeta(DataType::INT64, d_indices.dims(), d_indices.layout()));

  DenseTensor non_zero_elements(alloc,
                                DenseTensorMeta(d_non_zero_elements.dtype(),
                                                d_non_zero_elements.dims(),
                                                d_non_zero_elements.layout()));
  pten::Copy(*dev_ctx, d_indices, true, &indices);
  pten::Copy(*dev_ctx, d_non_zero_elements, true, &non_zero_elements);

  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(non_zero_elements.data<float>()[i], non_zero_data[i]);
    ASSERT_EQ(indices.data<int64_t>()[i], indices_data[i]);
    ASSERT_EQ(indices.data<int64_t>()[non_zero_num + i],
              indices_data[i + non_zero_num]);
  }
}

TEST(DEV_API, to_sparse_coo_hybird_cuda) {
  return;
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());

  DenseTensor dense_x(
      alloc,
      DenseTensorMeta(
          DataType::FLOAT32, framework::make_ddim({3, 3}), DataLayout::NCHW));

  DenseTensor d_dense_x(
      cuda_alloc,
      DenseTensorMeta(
          DataType::FLOAT32, framework::make_ddim({3, 3}), DataLayout::NCHW));

  auto* dense_x_data = dense_x.mutable_data<float>();
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {0.0, 1.0, 0.0, 3.2, 0.0, 0.0};
  std::vector<int64_t> indices_data = {0, 2};

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx_cuda = pool.Get(paddle::platform::CUDAPlace());
  auto* dev_ctx =
      static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda);

  pten::Copy(*dev_ctx, dense_x, true, &d_dense_x);

  // 2. test API
  auto sparse_out = DenseToSparseCoo<float>(
      *(static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda)),
      d_dense_x,
      1);

  int64_t non_zero_num = sparse_out.nnz();
  ASSERT_EQ(static_cast<uint64_t>(non_zero_num), indices_data.size());

  const auto& d_indices = sparse_out.non_zero_indices();
  const auto& d_non_zero_elements = sparse_out.non_zero_elements();
  DenseTensor indices(
      alloc,
      DenseTensorMeta(DataType::INT64, d_indices.dims(), d_indices.layout()));

  DenseTensor non_zero_elements(alloc,
                                DenseTensorMeta(d_non_zero_elements.dtype(),
                                                d_non_zero_elements.dims(),
                                                d_non_zero_elements.layout()));
  pten::Copy(*dev_ctx, d_indices, true, &indices);
  pten::Copy(*dev_ctx, d_non_zero_elements, true, &non_zero_elements);

  int cmp_elements = memcmp(non_zero_elements.data<float>(),
                            non_zero_data.data(),
                            non_zero_data.size() * sizeof(float));
  int cmp_indices = memcmp(indices.data<int64_t>(),
                           indices_data.data(),
                           sizeof(int64_t) * indices_data.size());
  ASSERT_EQ(cmp_elements, 0);
  ASSERT_EQ(cmp_indices, 0);
}

TEST(DEV_API, to_sparse_coo_performance) {
  return;
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());

  const int rows = 4096;
  const int cols = 4096;
  DenseTensor dense_x(alloc,
                      DenseTensorMeta(DataType::FLOAT32,
                                      framework::make_ddim({rows, cols}),
                                      DataLayout::NCHW));

  DenseTensor d_dense_x(cuda_alloc,
                        DenseTensorMeta(DataType::FLOAT32,
                                        framework::make_ddim({rows, cols}),
                                        DataLayout::NCHW));

  auto* dense_x_data = dense_x.mutable_data<float>();
  std::vector<float> dense_data(rows * cols);
  std::vector<float> non_zero_data;
  std::vector<int64_t> rows_data, cols_data;

  const float zero_rate = 0.9;
  std::default_random_engine random(time(NULL));
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      bool iszero = dis(random) < zero_rate;
      if (iszero) {
        dense_data[i * cols + j] = 0.0;
      } else {
        float data = dis(random);
        dense_data[i * cols + j] = data;
        non_zero_data.push_back(data);
        rows_data.push_back(i);
        cols_data.push_back(j);
      }
    }
  }

  std::copy(
      dense_data.data(), dense_data.data() + dense_data.size(), dense_x_data);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx_cuda = pool.Get(paddle::platform::CUDAPlace());
  auto* dev_ctx =
      static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda);

  pten::Copy(*dev_ctx, dense_x, true, &d_dense_x);

  // 2. test API
  auto sparse_out = DenseToSparseCoo<float>(
      *(static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda)),
      d_dense_x,
      2);

  int64_t non_zero_num = sparse_out.nnz();
  ASSERT_EQ(static_cast<uint64_t>(non_zero_num), rows_data.size());

  const auto& d_indices = sparse_out.non_zero_indices();
  const auto& d_non_zero_elements = sparse_out.non_zero_elements();
  DenseTensor indices(
      alloc,
      DenseTensorMeta(DataType::INT64, d_indices.dims(), d_indices.layout()));

  DenseTensor non_zero_elements(alloc,
                                DenseTensorMeta(d_non_zero_elements.dtype(),
                                                d_non_zero_elements.dims(),
                                                d_non_zero_elements.layout()));
  pten::Copy(*dev_ctx, d_indices, true, &indices);
  pten::Copy(*dev_ctx, d_non_zero_elements, true, &non_zero_elements);

  for (int64_t i = 0; i < non_zero_num; i++) {
    ASSERT_EQ(non_zero_elements.data<float>()[i], non_zero_data[i]);
    ASSERT_EQ(indices.data<int64_t>()[i], rows_data[i]);
    ASSERT_EQ(indices.data<int64_t>()[non_zero_num + i], cols_data[i]);
  }
}

}  // namespace tests
}  // namespace pten
