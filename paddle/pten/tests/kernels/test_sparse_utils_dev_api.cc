/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/pten/kernels/sparse/sparse_utils_kernel.h"

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

#if defined(PADDLE_WITH_CUDA)
  if (coo.place() == paddle::platform::CUDAPlace()) {
    const auto* dev_ctx_cuda =
        static_cast<const paddle::platform::CUDADeviceContext*>(dev_ctx);
    DenseTensor indices(
        alloc.get(),
        DenseTensorMeta(
            DataType::INT64, real_indices.dims(), real_indices.layout()));

    DenseTensor elements(alloc.get(),
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
#endif
    int cmp_indices = memcmp(real_indices.data<IndicesT>(),
                             non_zero_indices.data(),
                             non_zero_indices.size() * sizeof(IndicesT));
    ASSERT_EQ(cmp_indices, 0);
    int cmp_elements = memcmp(real_elements.data<ValueT>(),
                              non_zero_elements.data(),
                              non_zero_elements.size() * sizeof(ValueT));
    ASSERT_EQ(cmp_elements, 0);
#if defined(PADDLE_WITH_CUDA)
  }
#endif
}

template <typename T>
void TestDenseToSparseCoo(const DenseTensor& dense_x,
                          const int64_t sparse_dim,
                          const std::vector<T>& non_zero_data,
                          const std::vector<int64_t>& indices_data,
                          const int64_t non_zero_num) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  pten::CPUContext dev_ctx_cpu;
  // 1. test cpu
  auto cpu_sparse_out =
      sparse::DenseToSparseCoo<T>(dev_ctx_cpu, dense_x, sparse_dim);
  CheckResult<T, int64_t>(&dev_ctx_cpu,
                          cpu_sparse_out,
                          non_zero_data,
                          indices_data,
                          non_zero_num,
                          alloc);

// 2. test cuda
#if defined(PADDLE_WITH_CUDA)
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx_cuda = pool.GetByPlace(paddle::platform::CUDAPlace());
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  DenseTensor d_dense_x(
      cuda_alloc.get(),
      DenseTensorMeta(dense_x.dtype(), dense_x.dims(), dense_x.layout()));

  pten::Copy(*dev_ctx_cuda, dense_x, true, &d_dense_x);
  auto sparse_out =
      sparse::DenseToSparseCoo<T>(*dev_ctx_cuda, d_dense_x, sparse_dim);
  CheckResult<T, int64_t>(dev_ctx_cuda,
                          sparse_out,
                          non_zero_data,
                          indices_data,
                          non_zero_num,
                          alloc);
#endif
}

TEST(DEV_API, to_sparse_coo) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  std::default_random_engine random(time(NULL));
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  std::uniform_int_distribution<int> dis_int(4, 64);
  const int rows = dis_int(random), cols = dis_int(random);
  DenseTensor dense_x(
      alloc.get(),
      DenseTensorMeta(DataType::FLOAT32, {rows, cols}, DataLayout::NCHW));

  pten::CPUPlace cpu;
  auto* dense_x_data = dense_x.mutable_data<float>(cpu);
  std::vector<float> dense_data(rows * cols);
  std::vector<float> non_zero_data;
  std::vector<int64_t> rows_data, cols_data;
  const int64_t sparse_dim = 2;

  const float zero_rate = dis(random);

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

  std::vector<int64_t> indices_data(non_zero_num * 2);
  memcpy(&indices_data[0], &rows_data[0], non_zero_num * sizeof(int64_t));
  memcpy(&indices_data[non_zero_num],
         &cols_data[0],
         non_zero_num * sizeof(int64_t));

  TestDenseToSparseCoo(
      dense_x, sparse_dim, non_zero_data, indices_data, non_zero_num);
}

TEST(DEV_API, to_sparse_coo_hybird) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  DenseTensor dense_x(
      alloc.get(),
      DenseTensorMeta(DataType::FLOAT32, {3, 3}, DataLayout::NCHW));

  pten::CPUPlace cpu;
  const int64_t sparse_dim = 1;  // the non zero element is a vector
  auto* dense_x_data = dense_x.mutable_data<float>(cpu);
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {
      /*element0(*/ 0.0, 1.0, 0.0 /*)*/, /*element1(*/ 3.2, 0.0, 0.0 /*)*/};
  std::vector<int64_t> indices_data = {0, 2};
  const int64_t non_zero_num = 2;

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);
  TestDenseToSparseCoo(
      dense_x, sparse_dim, non_zero_data, indices_data, non_zero_num);
}

TEST(DEV_API, to_sparse_coo_fp16) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  DenseTensor dense_x(
      alloc.get(),
      DenseTensorMeta(DataType::FLOAT16, {3, 3}, DataLayout::NCHW));

  pten::CPUPlace cpu;
  const int64_t sparse_dim = 2;
  const int64_t non_zero_num = 2;
  auto* dense_x_data = dense_x.mutable_data<pten::dtype::float16>(cpu);
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {3.2, 0.0, 0.0}};
  std::vector<float> data = {1.0, 3.2};
  std::vector<pten::dtype::float16> non_zero_data(non_zero_num);
  for (int i = 0; i < non_zero_num; i++) {
    non_zero_data[i] = static_cast<pten::dtype::float16>(data[i]);
  }
  std::vector<int64_t> indices_data = {0, 2, 1, 0};

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);
  TestDenseToSparseCoo<paddle::float16>(
      dense_x, sparse_dim, non_zero_data, indices_data, non_zero_num);
}

TEST(DEV_API, to_sparse_coo_batch) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  DenseTensor dense_x(
      alloc.get(),
      DenseTensorMeta(DataType::FLOAT32, {2, 3, 3}, DataLayout::NCHW));

  pten::CPUPlace cpu;
  const int64_t sparse_dim = 3;
  const int64_t non_zero_num = 4;
  auto* dense_x_data = dense_x.mutable_data<float>(cpu);
  float dense_data[2][3][3] = {
      {{0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {4.0, 0.0, 0.0}}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<int64_t> indices_data = {0, 0, 1, 1, 0, 2, 1, 2, 1, 0, 1, 0};
  /*
      0, 0, 1, 1,
      0, 2, 1, 2,
      1, 0, 1, 0
   */

  std::copy(&dense_data[0][0][0], &dense_data[0][0][0] + 18, dense_x_data);
  TestDenseToSparseCoo<float>(
      dense_x, sparse_dim, non_zero_data, indices_data, non_zero_num);
}

}  // namespace tests
}  // namespace pten
