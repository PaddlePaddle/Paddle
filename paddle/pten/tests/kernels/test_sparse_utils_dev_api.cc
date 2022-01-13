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

template <typename ValueT, typename IndicesT>
inline void CheckResult(
    const DeviceContext* dev_ctx,
    const SparseCooTensor& coo,
    const std::vector<ValueT> non_zero_elements,
    const std::vector<IndicesT>& non_zero_indices,
    const int64_t non_zero_num,
    const std::shared_ptr<paddle::experimental::DefaultAllocator>& alloc) {
  const DenseTensor real_indices = coo.non_zero_indices();
  const DenseTensor real_elements = coo.non_zero_elements();
  ASSERT_EQ(coo.nnz(), non_zero_num);

  if (coo.place() == paddle::platform::CUDAPlace()) {
    const auto* dev_ctx_cuda =
        static_cast<const paddle::platform::CUDADeviceContext*>(dev_ctx);
    DenseTensor indices(
        alloc,
        DenseTensorMeta(
            DataType::INT64, real_indices.dims(), real_indices.layout()));

    DenseTensor elements(alloc,
                         DenseTensorMeta(real_elements.dtype(),
                                         real_elements.dims(),
                                         real_elements.layout()));
    pten::Copy(*dev_ctx_cuda, real_indices, true, &indices);
    pten::Copy(*dev_ctx_cuda, real_elements, true, &elements);

    int cmp_indices = memcmp(indices.data<IndicesT>(),
                             non_zero_indices.data(),
                             non_zero_indices.size() * sizeof(IndicesT));
    ASSERT_EQ(cmp_indices, 0);
    int cmp_elements = memcmp(elements.data<ValueT>(),
                              non_zero_elements.data(),
                              non_zero_elements.size() * sizeof(ValueT));
    ASSERT_EQ(cmp_elements, 0);
  } else {
    int cmp_indices = memcmp(real_indices.data<IndicesT>(),
                             non_zero_indices.data(),
                             non_zero_indices.size() * sizeof(IndicesT));
    ASSERT_EQ(cmp_indices, 0);
    int cmp_elements = memcmp(real_elements.data<ValueT>(),
                              non_zero_elements.data(),
                              non_zero_elements.size() * sizeof(ValueT));
    ASSERT_EQ(cmp_elements, 0);
  }
}

TEST(DEV_API, to_sparse_coo) {
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

  const int64_t sparse_dim = 2;
  auto* dense_x_data = dense_x.mutable_data<float>();
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};
  const int64_t non_zero_num = 4;

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* cpu = pool.Get(paddle::platform::CPUPlace());
  auto* cuda = pool.Get(paddle::platform::CUDAPlace());
  auto* dev_ctx_cuda = static_cast<paddle::platform::CUDADeviceContext*>(cuda);
  auto* dev_ctx_cpu = static_cast<paddle::platform::CPUDeviceContext*>(cpu);

  // 1. test cpu
  auto cpu_sparse_out = DenseToSparseCoo<float>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx_cpu)),
      dense_x,
      sparse_dim);
  CheckResult<float, int64_t>(dev_ctx_cpu,
                              cpu_sparse_out,
                              non_zero_data,
                              indices_data,
                              non_zero_num,
                              alloc);

  // 2. test cuda
  pten::Copy(*dev_ctx_cuda, dense_x, true, &d_dense_x);
  auto sparse_out = DenseToSparseCoo<float>(
      *(static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda)),
      d_dense_x,
      sparse_dim);

  CheckResult<float, int64_t>(dev_ctx_cuda,
                              sparse_out,
                              non_zero_data,
                              indices_data,
                              non_zero_num,
                              alloc);
}

TEST(DEV_API, to_sparse_coo_hybird) {
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

  const int64_t sparse_dim = 1;
  auto* dense_x_data = dense_x.mutable_data<float>();
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {0.0, 1.0, 0.0, 3.2, 0.0, 0.0};
  std::vector<int64_t> indices_data = {0, 2};
  const int64_t non_zero_num = 2;

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* cpu = pool.Get(paddle::platform::CPUPlace());
  auto* cuda = pool.Get(paddle::platform::CUDAPlace());
  auto* dev_ctx_cuda = static_cast<paddle::platform::CUDADeviceContext*>(cuda);
  auto* dev_ctx_cpu = static_cast<paddle::platform::CPUDeviceContext*>(cpu);

  // 1. test cpu
  auto cpu_sparse_out = DenseToSparseCoo<float>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx_cpu)),
      dense_x,
      sparse_dim);
  CheckResult<float, int64_t>(dev_ctx_cpu,
                              cpu_sparse_out,
                              non_zero_data,
                              indices_data,
                              non_zero_num,
                              alloc);

  // 2. test API
  pten::Copy(*dev_ctx_cuda, dense_x, true, &d_dense_x);
  auto sparse_out = DenseToSparseCoo<float>(
      *(static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda)),
      d_dense_x,
      1);
  CheckResult<float, int64_t>(dev_ctx_cuda,
                              sparse_out,
                              non_zero_data,
                              indices_data,
                              non_zero_num,
                              alloc);
}

TEST(DEV_API, to_sparse_coo_performance) {
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
  const int64_t sparse_dim = 2;

  const float zero_rate = 0.9;
  std::default_random_engine random(time(NULL));
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  int64_t non_zero_num = 0;
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
        non_zero_num += 1;
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

  auto sparse_out = DenseToSparseCoo<float>(
      *(static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda)),
      d_dense_x,
      sparse_dim);
  for (int i = 0; i < 100; i++) {
    DenseToSparseCoo<float>(
        *(static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda)),
        d_dense_x,
        sparse_dim);
  }
  std::vector<int64_t> indices_data(non_zero_num * 2);
  memcpy(&indices_data[0], &rows_data[0], non_zero_num * sizeof(int64_t));
  memcpy(&indices_data[non_zero_num],
         &cols_data[0],
         non_zero_num * sizeof(int64_t));
  CheckResult<float, int64_t>(dev_ctx_cuda,
                              sparse_out,
                              non_zero_data,
                              indices_data,
                              non_zero_num,
                              alloc);
}

}  // namespace tests
}  // namespace pten
