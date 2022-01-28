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

template <typename ValueT, typename IndicesT>
inline void CheckCsrResult(
    const DeviceContext* dev_ctx,
    const SparseCsrTensor& csr,
    const std::vector<ValueT> non_zero_elements,
    const std::vector<IndicesT>& non_zero_crows,
    const std::vector<IndicesT>& non_zero_cols,
    const int64_t non_zero_num,
    const std::shared_ptr<paddle::experimental::DefaultAllocator>& alloc) {
  const DenseTensor real_crows = csr.non_zero_crows();
  const DenseTensor real_cols = csr.non_zero_cols();
  const DenseTensor real_elements = csr.non_zero_elements();
  ASSERT_EQ(csr.non_zero_cols().numel(), non_zero_num);

#if defined(PADDLE_WITH_CUDA)
  if (csr.place() == paddle::platform::CUDAPlace()) {
    const auto* dev_ctx_cuda =
        static_cast<const paddle::platform::CUDADeviceContext*>(dev_ctx);
    DenseTensor crows(
        alloc.get(),
        DenseTensorMeta(
            DataType::INT64, real_crows.dims(), real_crows.layout()));
    DenseTensor cols(
        alloc.get(),
        DenseTensorMeta(DataType::INT64, real_cols.dims(), real_cols.layout()));

    DenseTensor elements(alloc.get(),
                         DenseTensorMeta(real_elements.dtype(),
                                         real_elements.dims(),
                                         real_elements.layout()));
    pten::Copy(*dev_ctx_cuda, real_crows, true, &crows);
    pten::Copy(*dev_ctx_cuda, real_cols, true, &cols);
    pten::Copy(*dev_ctx_cuda, real_elements, true, &elements);

    int cmp_crows = memcmp(crows.data<IndicesT>(),
                           non_zero_crows.data(),
                           non_zero_crows.size() * sizeof(IndicesT));
    ASSERT_EQ(cmp_crows, 0);
    int cmp_cols = memcmp(cols.data<IndicesT>(),
                          non_zero_cols.data(),
                          non_zero_cols.size() * sizeof(IndicesT));
    ASSERT_EQ(cmp_cols, 0);
    int cmp_elements = memcmp(elements.data<ValueT>(),
                              non_zero_elements.data(),
                              non_zero_elements.size() * sizeof(ValueT));
    ASSERT_EQ(cmp_elements, 0);
  } else {
#endif
    int cmp_crows = memcmp(real_crows.data<IndicesT>(),
                           non_zero_crows.data(),
                           non_zero_crows.size() * sizeof(IndicesT));
    ASSERT_EQ(cmp_crows, 0);
    int cmp_cols = memcmp(real_cols.data<IndicesT>(),
                          non_zero_cols.data(),
                          non_zero_cols.size() * sizeof(IndicesT));
    ASSERT_EQ(cmp_cols, 0);
    int cmp_elements = memcmp(real_elements.data<ValueT>(),
                              non_zero_elements.data(),
                              non_zero_elements.size() * sizeof(ValueT));
    ASSERT_EQ(cmp_elements, 0);
#if defined(PADDLE_WITH_CUDA)
  }
#endif
}

TEST(DEV_API, to_sparse_coo) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  DenseTensor dense_x(
      alloc.get(),
      DenseTensorMeta(
          DataType::FLOAT32, framework::make_ddim({3, 3}), DataLayout::NCHW));

  pten::CPUPlace cpu;
  const int64_t sparse_dim = 2;
  auto* dense_x_data = dense_x.mutable_data<float>(cpu);
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};
  const int64_t non_zero_num = 4;

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  pten::CPUContext dev_ctx_cpu;

  // 1. test cpu
  auto cpu_sparse_out =
      DenseToSparseCoo<float>(dev_ctx_cpu, dense_x, sparse_dim);
  CheckResult<float, int64_t>(&dev_ctx_cpu,
                              cpu_sparse_out,
                              non_zero_data,
                              indices_data,
                              non_zero_num,
                              alloc);

// 2. test cuda
#if defined(PADDLE_WITH_CUDA)
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  DenseTensor d_dense_x(
      cuda_alloc.get(),
      DenseTensorMeta(
          DataType::FLOAT32, framework::make_ddim({3, 3}), DataLayout::NCHW));

  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx_cuda = pool.GetByPlace(paddle::platform::CUDAPlace());
  pten::Copy(*dev_ctx_cuda, dense_x, true, &d_dense_x);
  auto sparse_out =
      DenseToSparseCoo<float>(*dev_ctx_cuda, d_dense_x, sparse_dim);

  CheckResult<float, int64_t>(dev_ctx_cuda,
                              sparse_out,
                              non_zero_data,
                              indices_data,
                              non_zero_num,
                              alloc);
#endif
}

TEST(DEV_API, to_sparse_coo_hybird) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  DenseTensor dense_x(
      alloc.get(),
      DenseTensorMeta(
          DataType::FLOAT32, framework::make_ddim({3, 3}), DataLayout::NCHW));

  pten::CPUPlace cpu;
  const int64_t sparse_dim = 1;
  auto* dense_x_data = dense_x.mutable_data<float>(cpu);
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {0.0, 1.0, 0.0, 3.2, 0.0, 0.0};
  std::vector<int64_t> indices_data = {0, 2};
  const int64_t non_zero_num = 2;

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);

  pten::CPUContext dev_ctx_cpu;

  // 1. test cpu
  auto cpu_sparse_out =
      DenseToSparseCoo<float>(dev_ctx_cpu, dense_x, sparse_dim);
  CheckResult<float, int64_t>(&dev_ctx_cpu,
                              cpu_sparse_out,
                              non_zero_data,
                              indices_data,
                              non_zero_num,
                              alloc);

// 2. test cuda
#if defined(PADDLE_WITH_CUDA)
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* cuda = pool.Get(paddle::platform::CUDAPlace());
  auto* dev_ctx_cuda = static_cast<paddle::platform::CUDADeviceContext*>(cuda);
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  DenseTensor d_dense_x(
      cuda_alloc.get(),
      DenseTensorMeta(
          DataType::FLOAT32, framework::make_ddim({3, 3}), DataLayout::NCHW));

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
#endif
}

TEST(DEV_API, to_sparse_coo_performance) {
#if defined(PADDLE_WITH_CUDA)
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());

  const int rows = 4096;
  const int cols = 4096;
  DenseTensor dense_x(alloc.get(),
                      DenseTensorMeta(DataType::FLOAT32,
                                      framework::make_ddim({rows, cols}),
                                      DataLayout::NCHW));

  DenseTensor d_dense_x(cuda_alloc.get(),
                        DenseTensorMeta(DataType::FLOAT32,
                                        framework::make_ddim({rows, cols}),
                                        DataLayout::NCHW));

  pten::CPUPlace cpu;
  auto* dense_x_data = dense_x.mutable_data<float>(cpu);
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
#endif
}

TEST(DEV_API, sparse_coo_to_dense) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  const int rows = 3;
  const int cols = 3;
  const int non_zero_num = 4;
  const int sparse_dim = 2;
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};

  pten::CPUContext dev_ctx_cpu;

  DDim dense_dims = framework::make_ddim({rows, cols});
  DenseTensor dense_indices(
      alloc.get(),
      DenseTensorMeta(DataType::INT64,
                      framework::make_ddim({sparse_dim, non_zero_num}),
                      DataLayout::NCHW));
  std::vector<int64_t> dense_elements_vec;
  dense_elements_vec.push_back(non_zero_num);
  for (int64_t i = sparse_dim; i < dense_dims.size(); i++) {
    dense_elements_vec.push_back(dense_dims[i]);
  }
  DDim dense_elements_dims = framework::make_ddim(dense_elements_vec);
  DenseTensor dense_elements(
      alloc.get(),
      DenseTensorMeta(
          DataType::FLOAT32, dense_elements_dims, DataLayout::NCHW));

  pten::CPUPlace cpu_place;
  memcpy(dense_indices.mutable_data<int64_t>(cpu_place),
         indices_data.data(),
         indices_data.size() * sizeof(int64_t));
  memcpy(dense_elements.mutable_data<float>(cpu_place),
         non_zero_data.data(),
         non_zero_num * sizeof(float));

  SparseCooTensor coo(dense_indices, dense_elements, dense_dims);

  auto dense_out = SparseCooToDense<float>(dev_ctx_cpu, coo);

  int cmp = memcmp(
      &dense_data[0][0], dense_out.data<float>(), sizeof(float) * rows * cols);
  ASSERT_EQ(cmp, 0);

#if defined(PADDLE_WITH_CUDA)
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* cuda = pool.Get(paddle::platform::CUDAPlace());
  auto* dev_ctx_cuda = static_cast<paddle::platform::CUDADeviceContext*>(cuda);
  DenseTensor d_dense_indices(
      cuda_alloc.get(),
      DenseTensorMeta(DataType::INT64, dense_indices.dims(), DataLayout::NCHW));
  DenseTensor d_dense_elements(
      cuda_alloc.get(),
      DenseTensorMeta(
          DataType::FLOAT32, dense_elements_dims, DataLayout::NCHW));
  pten::Copy(*dev_ctx_cuda, dense_indices, true, &d_dense_indices);
  pten::Copy(*dev_ctx_cuda, dense_elements, true, &d_dense_elements);
  SparseCooTensor coo_cuda(d_dense_indices, d_dense_elements, dense_dims);
  auto dense_out_cuda = SparseCooToDense<float>(
      *(static_cast<paddle::platform::CUDADeviceContext*>(dev_ctx_cuda)),
      coo_cuda);

  DenseTensor h_dense_out(alloc.get(),
                          DenseTensorMeta(dense_out_cuda.dtype(),
                                          dense_out_cuda.dims(),
                                          dense_out_cuda.layout()));
  pten::Copy(*dev_ctx_cuda, dense_out_cuda, true, &h_dense_out);
  int cmp_cuda = memcmp(&dense_data[0][0],
                        h_dense_out.data<float>(),
                        sizeof(float) * rows * cols);
  ASSERT_EQ(cmp_cuda, 0);
#endif
}

TEST(DEV_API, sparse_csr_to_coo) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(alloc.get(),
                            DenseTensorMeta(pten::DataType::FLOAT32,
                                            framework::make_ddim({3, 3}),
                                            DataLayout::NCHW));
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const int64_t non_zero_num = 4;

  framework::DDim dense_dims = framework::make_ddim({3, 3});
  pten::DenseTensorMeta crows_meta(DataType::INT64,
                                   framework::make_ddim({dense_dims[0] + 1}),
                                   DataLayout::NCHW);
  pten::DenseTensorMeta cols_meta(
      DataType::INT64, framework::make_ddim({non_zero_num}), DataLayout::NCHW);
  pten::DenseTensorMeta values_meta(DataType::FLOAT32,
                                    framework::make_ddim({non_zero_num}),
                                    DataLayout::NCHW);

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
  pten::SparseCsrTensor csr(crows, cols, values, dense_dims);

  // 1. test cpu
  pten::CPUContext dev_ctx_cpu;
  auto cpu_sparse_out = SparseCsrToCoo<float>(dev_ctx_cpu, csr);
  CheckResult<float, int64_t>(&dev_ctx_cpu,
                              cpu_sparse_out,
                              non_zero_data,
                              indices_data,
                              non_zero_num,
                              alloc);

// 2. test cuda
#if defined(PADDLE_WITH_CUDA)
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx_cuda = pool.GetByPlace(paddle::platform::CUDAPlace());
  pten::DenseTensor d_crows(cuda_alloc.get(), crows_meta);
  pten::DenseTensor d_cols(cuda_alloc.get(), cols_meta);
  pten::DenseTensor d_values(cuda_alloc.get(), values_meta);
  pten::Copy(*dev_ctx_cuda, crows, true, &d_crows);
  pten::Copy(*dev_ctx_cuda, cols, true, &d_cols);
  pten::Copy(*dev_ctx_cuda, values, true, &d_values);
  pten::SparseCsrTensor d_csr(d_crows, d_cols, d_values, dense_dims);
  auto cuda_sparse_out = SparseCsrToCoo<float>(*dev_ctx_cuda, d_csr);
  CheckResult<float, int64_t>(dev_ctx_cuda,
                              cuda_sparse_out,
                              non_zero_data,
                              indices_data,
                              non_zero_num,
                              alloc);
#endif
}

template <typename T>
void TestDenseToSparseCsr(const DenseTensor& dense_x,
                          const int64_t non_zero_num,
                          const std::vector<T>& non_zero_data,
                          const std::vector<int64_t>& crows_data,
                          const std::vector<int64_t>& cols_data) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::CPUContext dev_ctx_cpu;

  // 1. test cpu
  auto cpu_sparse_out = DenseToSparseCsr<T>(dev_ctx_cpu, dense_x);
  CheckCsrResult<T, int64_t>(&dev_ctx_cpu,
                             cpu_sparse_out,
                             non_zero_data,
                             crows_data,
                             cols_data,
                             non_zero_num,
                             alloc);

// 2. test cuda
#if defined(PADDLE_WITH_CUDA)
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  DenseTensor d_dense_x(
      cuda_alloc.get(),
      DenseTensorMeta(dense_x.dtype(), dense_x.dims(), dense_x.layout()));

  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx_cuda = pool.GetByPlace(paddle::platform::CUDAPlace());
  pten::Copy(*dev_ctx_cuda, dense_x, true, &d_dense_x);
  auto sparse_out = DenseToSparseCsr<T>(*dev_ctx_cuda, d_dense_x);

  CheckCsrResult<T, int64_t>(dev_ctx_cuda,
                             sparse_out,
                             non_zero_data,
                             crows_data,
                             cols_data,
                             non_zero_num,
                             alloc);
#endif
}

TEST(DEV_API, dense_to_sparse_csr) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  DenseTensor dense_x(
      alloc.get(),
      DenseTensorMeta(
          DataType::FLOAT32, framework::make_ddim({3, 3}), DataLayout::NCHW));

  pten::CPUPlace cpu;
  auto* dense_x_data = dense_x.mutable_data<float>(cpu);
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const int64_t non_zero_num = 4;

  std::copy(&dense_data[0][0], &dense_data[0][0] + 9, dense_x_data);
  TestDenseToSparseCsr<float>(
      dense_x, non_zero_num, non_zero_data, crows_data, cols_data);
}

TEST(DEV_API, dense_to_sparse_csr_batch) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  DenseTensor dense_x(alloc.get(),
                      DenseTensorMeta(DataType::FLOAT16,
                                      framework::make_ddim({2, 3, 3}),
                                      DataLayout::NCHW));

  pten::CPUPlace cpu;
  auto* dense_x_data = dense_x.mutable_data<pten::dtype::float16>(cpu);
  const int64_t non_zero_num = 7;
  float dense_data[2][3][3] = {
      {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}},
      {{0.0, 1.0, 0.0}, {2.0, 0.0, 0.0}, {3.2, 0.0, 0.0}}};
  std::vector<float> data = {1.0, 2.0, 3.0, 3.2, 1.0, 2.0, 3.2};
  std::vector<pten::dtype::float16> non_zero_data(non_zero_num);
  for (int64_t i = 0; i < non_zero_num; i++) {
    non_zero_data[i] = static_cast<pten::dtype::float16>(data[i]);
  }
  std::vector<int64_t> cols_data = {1, 0, 2, 0, 1, 0, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4, 0, 1, 2, 3};

  std::copy(&dense_data[0][0][0], &dense_data[0][0][0] + 9, dense_x_data);
  TestDenseToSparseCsr<pten::dtype::float16>(
      dense_x, non_zero_num, non_zero_data, crows_data, cols_data);
}

template <typename T>
void TestCooToCsr(const DDim& dense_dims,
                  const int64_t& non_zero_num,
                  const std::vector<T>& non_zero_data,
                  const std::vector<int64_t>& non_zero_indices,
                  const std::vector<int64_t>& cols_data,
                  const std::vector<int64_t>& crows_data) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  pten::CPUPlace cpu;
  DenseTensorMeta indices_meta(
      DataType::INT64,
      {static_cast<int64_t>(dense_dims.size()), non_zero_num},
      DataLayout::NCHW);
  DenseTensor indices(alloc.get(), indices_meta);
  DenseTensorMeta values_meta(
      paddle::experimental::CppTypeToDataType<T>::Type(),
      {non_zero_num},
      DataLayout::NCHW);
  DenseTensor values(alloc.get(), values_meta);

  memcpy(indices.mutable_data<int64_t>(cpu),
         non_zero_indices.data(),
         non_zero_indices.size() * sizeof(int64_t));
  memcpy(values.mutable_data<T>(cpu),
         non_zero_data.data(),
         non_zero_data.size() * sizeof(T));
  pten::SparseCooTensor coo(indices, values, dense_dims);

  // 1. test cpu
  pten::CPUContext dev_ctx_cpu;
  auto cpu_sparse_out = SparseCooToCsr<T>(dev_ctx_cpu, coo);
  CheckCsrResult<T, int64_t>(&dev_ctx_cpu,
                             cpu_sparse_out,
                             non_zero_data,
                             crows_data,
                             cols_data,
                             non_zero_num,
                             alloc);

// 2. test cuda
#if defined(PADDLE_WITH_CUDA)
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx_cuda = pool.GetByPlace(paddle::platform::CUDAPlace());
  pten::DenseTensor d_indices(cuda_alloc.get(), indices_meta);
  pten::DenseTensor d_values(cuda_alloc.get(), values_meta);
  pten::Copy(*dev_ctx_cuda, indices, true, &d_indices);
  pten::Copy(*dev_ctx_cuda, values, true, &d_values);
  pten::SparseCooTensor d_coo(d_indices, d_values, dense_dims);
  auto cuda_sparse_out = SparseCooToCsr<T>(*dev_ctx_cuda, d_coo);
  CheckCsrResult<T, int64_t>(dev_ctx_cuda,
                             cuda_sparse_out,
                             non_zero_data,
                             crows_data,
                             cols_data,
                             non_zero_num,
                             alloc);
#endif
}

TEST(DEV_API, coo_to_csr) {
  // float dense_data[3][3] = {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0,
  // 0.0}};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> non_zero_indices = {0, 1, 1, 2, 1, 0, 2, 0};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const int64_t non_zero_num = 4;
  auto dense_dims = pten::framework::make_ddim({3, 3});
  TestCooToCsr<float>(dense_dims,
                      non_zero_num,
                      non_zero_data,
                      non_zero_indices,
                      cols_data,
                      crows_data);
}

TEST(DEV_API, batch_coo_to_csr) {
  // float dense_data[2][3][3] =
  //  {{{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}},
  //  {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {0.0, 0.0, 0.0}}};
  const int64_t non_zero_num = 7;
  std::vector<float> data = {1.0, 2.0, 3.0, 3.2, 1.0, 2.0, 3.0};
  std::vector<pten::dtype::float16> non_zero_data(non_zero_num);
  for (int64_t i = 0; i < non_zero_num; i++) {
    non_zero_data[i] = static_cast<pten::dtype::float16>(data[i]);
  }
  std::vector<int64_t> non_zero_indices = {0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 2,
                                           0, 1, 1, 1, 0, 2, 0, 1, 0, 2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0, 1, 0, 2};
  std::vector<int64_t> crows_data = {0, 1, 3, 4, 0, 1, 3, 3};
  auto dense_dims = pten::framework::make_ddim({2, 3, 3});
  TestCooToCsr<pten::dtype::float16>(dense_dims,
                                     non_zero_num,
                                     non_zero_data,
                                     non_zero_indices,
                                     cols_data,
                                     crows_data);
}

}  // namespace tests
}  // namespace pten
