/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <memory>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"

namespace phi {
namespace tests {

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
  if (coo.place() == phi::GPUPlace()) {
    const auto* dev_ctx_gpu = static_cast<const phi::GPUContext*>(dev_ctx);
    DenseTensor indices(
        alloc.get(),
        DenseTensorMeta(
            DataType::INT64, real_indices.dims(), real_indices.layout()));

    DenseTensor elements(alloc.get(),
                         DenseTensorMeta(real_elements.dtype(),
                                         real_elements.dims(),
                                         real_elements.layout()));
    phi::Copy(*dev_ctx_gpu, real_indices, indices.place(), true, &indices);
    phi::Copy(*dev_ctx_gpu, real_elements, elements.place(), true, &elements);

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

  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());

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
  phi::GPUContext dev_ctx_gpu;
  dev_ctx_gpu.PartialInitWithoutAllocator();
  dev_ctx_gpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(dev_ctx_gpu.GetPlace(), dev_ctx_gpu.stream())
          .get());
  dev_ctx_gpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx_gpu.PartialInitWithAllocator();

  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  DenseTensor d_dense_x(
      cuda_alloc.get(),
      DenseTensorMeta(dense_x.dtype(), dense_x.dims(), dense_x.layout()));

  phi::Copy(dev_ctx_gpu, dense_x, phi::GPUPlace(), true, &d_dense_x);
  auto sparse_out =
      sparse::DenseToSparseCoo<T>(dev_ctx_gpu, d_dense_x, sparse_dim);
  CheckResult<T, int64_t>(&dev_ctx_gpu,
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

  phi::CPUPlace cpu;
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

  phi::CPUPlace cpu;
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

  phi::CPUPlace cpu;
  const int64_t sparse_dim = 2;
  const int64_t non_zero_num = 2;
  auto* dense_x_data = dense_x.mutable_data<phi::dtype::float16>(cpu);
  float dense_data[3][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {3.2, 0.0, 0.0}};
  std::vector<float> data = {1.0, 3.2};
  std::vector<phi::dtype::float16> non_zero_data(non_zero_num);
  for (int i = 0; i < non_zero_num; i++) {
    non_zero_data[i] = static_cast<phi::dtype::float16>(data[i]);
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

  phi::CPUPlace cpu;
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

template <typename T>
void TestSparseCsrToCoo(const DDim& dense_dims,
                        const std::vector<T>& non_zero_data,
                        const std::vector<int64_t>& crows_data,
                        const std::vector<int64_t>& cols_data,
                        const std::vector<int64_t>& indices_data,
                        const int64_t non_zero_num) {
  int batchs = 1;
  int rows = dense_dims[0];
  if (dense_dims.size() == 3) {
    batchs = dense_dims[0];
    rows = dense_dims[1];
  }
  phi::DenseTensorMeta crows_meta(
      DataType::INT64, {batchs * (rows + 1)}, DataLayout::NCHW);
  phi::DenseTensorMeta cols_meta(
      DataType::INT64, {non_zero_num}, DataLayout::NCHW);
  phi::DenseTensorMeta values_meta(
      paddle::experimental::CppTypeToDataType<T>::Type(),
      {non_zero_num},
      DataLayout::NCHW);
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  phi::CPUPlace place;
  phi::DenseTensor crows(alloc.get(), crows_meta);
  phi::DenseTensor cols(alloc.get(), cols_meta);
  phi::DenseTensor values(alloc.get(), values_meta);
  memcpy(crows.mutable_data<int64_t>(place),
         crows_data.data(),
         crows_data.size() * sizeof(int64_t));
  memcpy(cols.mutable_data<int64_t>(place),
         cols_data.data(),
         cols_data.size() * sizeof(int64_t));
  memcpy(values.mutable_data<T>(place),
         non_zero_data.data(),
         non_zero_data.size() * sizeof(T));
  phi::SparseCsrTensor csr(crows, cols, values, dense_dims);

  // 1. test cpu
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  auto cpu_sparse_out = sparse::SparseCsrToCoo<T>(dev_ctx_cpu, csr);
  CheckResult<T, int64_t>(&dev_ctx_cpu,
                          cpu_sparse_out,
                          non_zero_data,
                          indices_data,
                          non_zero_num,
                          alloc);
// 2. test cuda
#if defined(PADDLE_WITH_CUDA)
  phi::GPUContext dev_ctx_gpu;
  dev_ctx_gpu.PartialInitWithoutAllocator();
  dev_ctx_gpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(dev_ctx_gpu.GetPlace(), dev_ctx_gpu.stream())
          .get());
  dev_ctx_gpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx_gpu.PartialInitWithAllocator();

  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  phi::DenseTensor d_crows(cuda_alloc.get(), crows_meta);
  phi::DenseTensor d_cols(cuda_alloc.get(), cols_meta);
  phi::DenseTensor d_values(cuda_alloc.get(), values_meta);
  phi::Copy(dev_ctx_gpu, crows, d_crows.place(), true, &d_crows);
  phi::Copy(dev_ctx_gpu, cols, d_cols.place(), true, &d_cols);
  phi::Copy(dev_ctx_gpu, values, d_values.place(), true, &d_values);
  phi::SparseCsrTensor d_csr(d_crows, d_cols, d_values, dense_dims);
  auto cuda_sparse_out = sparse::SparseCsrToCoo<T>(dev_ctx_gpu, d_csr);
  CheckResult<T, int64_t>(&dev_ctx_gpu,
                          cuda_sparse_out,
                          non_zero_data,
                          indices_data,
                          non_zero_num,
                          alloc);
#endif
}

TEST(DEV_API, sparse_csr_to_coo) {
  DDim dense_dims = phi::make_ddim({3, 3});
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const int64_t non_zero_num = 4;
  TestSparseCsrToCoo(dense_dims,
                     non_zero_data,
                     crows_data,
                     cols_data,
                     indices_data,
                     non_zero_num);
}

TEST(DEV_API, sparse_csr_to_coo_batch_and_fp16) {
  DDim dense_dims = phi::make_ddim({2, 3, 3});
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2, 1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0, 1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4, 0, 1, 3, 4};
  std::vector<int64_t> indices_data = {0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 2,
                                       0, 1, 1, 2, 1, 0, 2, 0, 1, 0, 2, 0};
  const int64_t non_zero_num = 8;
  using float16 = phi::dtype::float16;
  std::vector<float16> non_zero_data_fp16(non_zero_num);
  for (int64_t i = 0; i < non_zero_num; i++) {
    non_zero_data_fp16[i] = static_cast<float16>(non_zero_data[i]);
  }
  TestSparseCsrToCoo(dense_dims,
                     non_zero_data_fp16,
                     crows_data,
                     cols_data,
                     indices_data,
                     non_zero_num);
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
    const auto* dev_ctx_gpu = static_cast<const phi::GPUContext*>(dev_ctx);
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
    phi::Copy(*dev_ctx_gpu, real_crows, crows.place(), true, &crows);
    phi::Copy(*dev_ctx_gpu, real_cols, cols.place(), true, &cols);
    phi::Copy(*dev_ctx_gpu, real_elements, elements.place(), true, &elements);

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

template <typename T>
void TestCooToCsr(const DDim& dense_dims,
                  const int64_t& non_zero_num,
                  const std::vector<T>& non_zero_data,
                  const std::vector<int64_t>& non_zero_indices,
                  const std::vector<int64_t>& cols_data,
                  const std::vector<int64_t>& crows_data) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  phi::CPUPlace cpu;
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
  phi::SparseCooTensor coo(indices, values, dense_dims);

  // 1. test cpu
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  auto cpu_sparse_out = sparse::SparseCooToCsr<T>(dev_ctx_cpu, coo);
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
  phi::GPUContext dev_ctx_gpu;
  dev_ctx_gpu.PartialInitWithoutAllocator();
  dev_ctx_gpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(dev_ctx_gpu.GetPlace(), dev_ctx_gpu.stream())
          .get());
  dev_ctx_gpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx_gpu.PartialInitWithAllocator();
  phi::DenseTensor d_indices(cuda_alloc.get(), indices_meta);
  phi::DenseTensor d_values(cuda_alloc.get(), values_meta);
  phi::Copy(dev_ctx_gpu, indices, phi::GPUPlace(), true, &d_indices);
  phi::Copy(dev_ctx_gpu, values, phi::GPUPlace(), true, &d_values);
  phi::SparseCooTensor d_coo(d_indices, d_values, dense_dims);
  auto cuda_sparse_out = sparse::SparseCooToCsr<T>(dev_ctx_gpu, d_coo);
  CheckCsrResult<T, int64_t>(&dev_ctx_gpu,
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
  auto dense_dims = phi::make_ddim({3, 3});
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
  std::vector<phi::dtype::float16> non_zero_data(non_zero_num);
  for (int64_t i = 0; i < non_zero_num; i++) {
    non_zero_data[i] = static_cast<phi::dtype::float16>(data[i]);
  }
  std::vector<int64_t> non_zero_indices = {0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 2,
                                           0, 1, 1, 1, 0, 2, 0, 1, 0, 2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0, 1, 0, 2};
  std::vector<int64_t> crows_data = {0, 1, 3, 4, 0, 1, 3, 3};
  auto dense_dims = phi::make_ddim({2, 3, 3});
  TestCooToCsr<phi::dtype::float16>(dense_dims,
                                    non_zero_num,
                                    non_zero_data,
                                    non_zero_indices,
                                    cols_data,
                                    crows_data);
}

template <typename T>
void TestDenseToSparseCsr(const DenseTensor& dense_x,
                          const int64_t non_zero_num,
                          const std::vector<T>& non_zero_data,
                          const std::vector<int64_t>& crows_data,
                          const std::vector<int64_t>& cols_data) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());

  // 1. test cpu
  auto cpu_sparse_out = sparse::DenseToSparseCsr<T>(dev_ctx_cpu, dense_x);
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

  phi::GPUContext dev_ctx_gpu;
  dev_ctx_gpu.PartialInitWithoutAllocator();
  dev_ctx_gpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(dev_ctx_gpu.GetPlace(), dev_ctx_gpu.stream())
          .get());
  dev_ctx_gpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx_gpu.PartialInitWithAllocator();
  phi::Copy(dev_ctx_gpu, dense_x, phi::GPUPlace(), true, &d_dense_x);
  auto sparse_out = sparse::DenseToSparseCsr<T>(dev_ctx_gpu, d_dense_x);

  CheckCsrResult<T, int64_t>(&dev_ctx_gpu,
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
          DataType::FLOAT32, phi::make_ddim({3, 3}), DataLayout::NCHW));

  phi::CPUPlace cpu;
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

  DenseTensor dense_x(
      alloc.get(),
      DenseTensorMeta(
          DataType::FLOAT16, phi::make_ddim({2, 3, 3}), DataLayout::NCHW));

  phi::CPUPlace cpu;
  auto* dense_x_data = dense_x.mutable_data<phi::dtype::float16>(cpu);
  const int64_t non_zero_num = 7;
  float dense_data[2][3][3] = {
      {{0.0, 1.0, 0.0}, {2.0, 0.0, 3.0}, {3.2, 0.0, 0.0}},
      {{0.0, 1.0, 0.0}, {2.0, 0.0, 0.0}, {3.2, 0.0, 0.0}}};
  std::vector<float> data = {1.0, 2.0, 3.0, 3.2, 1.0, 2.0, 3.2};
  std::vector<phi::dtype::float16> non_zero_data(non_zero_num);
  for (int64_t i = 0; i < non_zero_num; i++) {
    non_zero_data[i] = static_cast<phi::dtype::float16>(data[i]);
  }
  std::vector<int64_t> cols_data = {1, 0, 2, 0, 1, 0, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4, 0, 1, 2, 3};

  float* dense_ptr = &dense_data[0][0][0];
  for (int i = 0; i < 18; i++) {
    dense_x_data[i] = static_cast<phi::dtype::float16>(dense_ptr[i]);
  }
  TestDenseToSparseCsr<phi::dtype::float16>(
      dense_x, non_zero_num, non_zero_data, crows_data, cols_data);
}

template <typename T>
void TestSparseCooToDense(const DDim& dense_dims,
                          const std::vector<T>& dense_data,
                          const std::vector<T>& non_zero_data,
                          const std::vector<int64_t>& indices_data,
                          const int64_t non_zero_num,
                          const int64_t sparse_dim) {
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  DenseTensor dense_indices(
      alloc.get(),
      DenseTensorMeta(DataType::INT64,
                      phi::make_ddim({sparse_dim, non_zero_num}),
                      DataLayout::NCHW));
  std::vector<int64_t> dense_elements_vec;
  dense_elements_vec.push_back(non_zero_num);
  for (int64_t i = sparse_dim; i < dense_dims.size(); i++) {
    dense_elements_vec.push_back(dense_dims[i]);
  }
  DDim dense_elements_dims = phi::make_ddim(dense_elements_vec);
  DenseTensor dense_elements(
      alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      dense_elements_dims,
                      DataLayout::NCHW));

  phi::CPUPlace cpu_place;
  memcpy(dense_indices.mutable_data<int64_t>(cpu_place),
         indices_data.data(),
         indices_data.size() * sizeof(int64_t));
  memcpy(dense_elements.mutable_data<T>(cpu_place),
         non_zero_data.data(),
         non_zero_num * sizeof(T));

  SparseCooTensor coo(dense_indices, dense_elements, dense_dims);

  DenseTensor dense_out = sparse::SparseCooToDense<T>(dev_ctx_cpu, coo);

  int cmp = memcmp(
      &dense_data[0], dense_out.data<T>(), sizeof(T) * dense_data.size());
  ASSERT_EQ(cmp, 0);

#if defined(PADDLE_WITH_CUDA)
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  phi::GPUContext dev_ctx_gpu;
  dev_ctx_gpu.PartialInitWithoutAllocator();
  dev_ctx_gpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(dev_ctx_gpu.GetPlace(), dev_ctx_gpu.stream())
          .get());
  dev_ctx_gpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx_gpu.PartialInitWithAllocator();
  DenseTensor d_dense_indices(cuda_alloc.get(), dense_indices.meta());
  DenseTensor d_dense_elements(cuda_alloc.get(), dense_elements.meta());
  phi::Copy(
      dev_ctx_gpu, dense_indices, phi::GPUPlace(), true, &d_dense_indices);
  phi::Copy(
      dev_ctx_gpu, dense_elements, phi::GPUPlace(), true, &d_dense_elements);
  SparseCooTensor coo_cuda(d_dense_indices, d_dense_elements, dense_dims);
  auto dense_out_cuda = sparse::SparseCooToDense<T>(dev_ctx_gpu, coo_cuda);

  DenseTensor h_dense_out(alloc.get(),
                          DenseTensorMeta(dense_out_cuda.dtype(),
                                          dense_out_cuda.dims(),
                                          dense_out_cuda.layout()));
  phi::Copy(
      dev_ctx_gpu, dense_out_cuda, h_dense_out.place(), true, &h_dense_out);
  int cmp_cuda = memcmp(
      &dense_data[0], h_dense_out.data<T>(), sizeof(T) * dense_data.size());
  ASSERT_EQ(cmp_cuda, 0);
#endif
}

TEST(DEV_API, sparse_coo_to_dense) {
  const int non_zero_num = 4;
  const int sparse_dim = 2;
  std::vector<float> dense_data = {0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0, 0.0};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> indices_data = {0, 1, 1, 2, 1, 0, 2, 0};
  DDim dense_dims = phi::make_ddim({3, 3});
  TestSparseCooToDense(dense_dims,
                       dense_data,
                       non_zero_data,
                       indices_data,
                       non_zero_num,
                       sparse_dim);
}

TEST(DEV_API, sparse_coo_to_dense_batch_and_fp16) {
  std::vector<float> dense_data = {0.0,
                                   1.0,
                                   0.0,
                                   0.0,
                                   0.0,
                                   0.0,
                                   2.0,
                                   0.0,
                                   0.0,
                                   0.0,
                                   0.0,
                                   0.0,
                                   0.0,
                                   3.0,
                                   0.0,
                                   4.0,
                                   0.0,
                                   0.0};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<int64_t> indices_data = {0, 0, 1, 1, 0, 2, 1, 2, 1, 0, 1, 0};
  const int non_zero_num = 4;
  const int sparse_dim = 3;
  DDim dense_dims = phi::make_ddim({2, 3, 3});
  using float16 = phi::dtype::float16;
  std::vector<float16> dense_data_fp16(dense_data.size()),
      non_zero_data_fp16(non_zero_num);
  for (uint64_t i = 0; i < dense_data.size(); i++) {
    dense_data_fp16[i] = static_cast<float16>(dense_data[i]);
  }
  for (int64_t i = 0; i < non_zero_num; i++) {
    non_zero_data_fp16[i] = static_cast<float16>(non_zero_data[i]);
  }
  TestSparseCooToDense(dense_dims,
                       dense_data_fp16,
                       non_zero_data_fp16,
                       indices_data,
                       non_zero_num,
                       sparse_dim);
}

template <typename T>
void TestSparseCsrToDense(const DDim& dense_dims,
                          const std::vector<T>& dense_data,
                          const std::vector<T>& non_zero_data,
                          const std::vector<int64_t>& crows_data,
                          const std::vector<int64_t>& cols_data,
                          const int64_t non_zero_num) {
  int batchs = 1;
  int rows = dense_dims[0];
  if (dense_dims.size() == 3) {
    batchs = dense_dims[0];
    rows = dense_dims[1];
  }
  phi::DenseTensorMeta crows_meta(
      DataType::INT64, phi::make_ddim({batchs * (rows + 1)}), DataLayout::NCHW);
  phi::DenseTensorMeta cols_meta(
      DataType::INT64, phi::make_ddim({non_zero_num}), DataLayout::NCHW);
  phi::DenseTensorMeta values_meta(
      paddle::experimental::CppTypeToDataType<T>::Type(),
      phi::make_ddim({non_zero_num}),
      DataLayout::NCHW);
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  phi::CPUPlace place;
  phi::DenseTensor crows(alloc.get(), crows_meta);
  phi::DenseTensor cols(alloc.get(), cols_meta);
  phi::DenseTensor values(alloc.get(), values_meta);
  memcpy(crows.mutable_data<int64_t>(place),
         crows_data.data(),
         crows_data.size() * sizeof(int64_t));
  memcpy(cols.mutable_data<int64_t>(place),
         cols_data.data(),
         cols_data.size() * sizeof(int64_t));
  memcpy(values.mutable_data<T>(place),
         non_zero_data.data(),
         non_zero_data.size() * sizeof(T));
  phi::SparseCsrTensor csr(crows, cols, values, dense_dims);

  // 1. test cpu
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  DenseTensor cpu_sparse_out = sparse::SparseCsrToDense<T>(dev_ctx_cpu, csr);
  int cmp_cpu = memcmp(cpu_sparse_out.data<T>(),
                       dense_data.data(),
                       sizeof(T) * dense_data.size());
  ASSERT_EQ(cmp_cpu, 0);

// 2. test cuda
#if defined(PADDLE_WITH_CUDA)
  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  phi::GPUContext dev_ctx_gpu;
  dev_ctx_gpu.PartialInitWithoutAllocator();
  dev_ctx_gpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(dev_ctx_gpu.GetPlace(), dev_ctx_gpu.stream())
          .get());
  dev_ctx_gpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx_gpu.PartialInitWithAllocator();
  phi::DenseTensor d_crows(cuda_alloc.get(), crows_meta);
  phi::DenseTensor d_cols(cuda_alloc.get(), cols_meta);
  phi::DenseTensor d_values(cuda_alloc.get(), values_meta);
  phi::Copy(dev_ctx_gpu, crows, phi::GPUPlace(), true, &d_crows);
  phi::Copy(dev_ctx_gpu, cols, phi::GPUPlace(), true, &d_cols);
  phi::Copy(dev_ctx_gpu, values, phi::GPUPlace(), true, &d_values);
  phi::SparseCsrTensor d_csr(d_crows, d_cols, d_values, dense_dims);
  auto cuda_sparse_out = sparse::SparseCsrToDense<T>(dev_ctx_gpu, d_csr);
  phi::DenseTensor h_out(alloc.get(), cpu_sparse_out.meta());
  phi::Copy(dev_ctx_gpu, cuda_sparse_out, phi::CPUPlace(), true, &h_out);
  int cmp_cuda =
      memcmp(h_out.data<T>(), dense_data.data(), sizeof(T) * dense_data.size());
  ASSERT_EQ(cmp_cuda, 0);
#endif
}

TEST(DEV_API, sparse_csr_to_dense) {
  DDim dense_dims = phi::make_ddim({3, 3});
  std::vector<float> dense_data = {0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0, 0.0};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4};
  const int64_t non_zero_num = 4;

  TestSparseCsrToDense(dense_dims,
                       dense_data,
                       non_zero_data,
                       crows_data,
                       cols_data,
                       non_zero_num);
}

TEST(DEV_API, sparse_csr_to_dense_batch_and_fp16) {
  DDim dense_dims = phi::make_ddim({2, 3, 3});
  std::vector<float> dense_data = {0.0,
                                   1.0,
                                   0.0,
                                   2.0,
                                   0.0,
                                   3.0,
                                   3.2,
                                   0.0,
                                   0.0,
                                   0.0,
                                   1.0,
                                   0.0,
                                   2.0,
                                   0.0,
                                   3.0,
                                   3.2,
                                   0.0,
                                   0.0};
  std::vector<float> non_zero_data = {1.0, 2.0, 3.0, 3.2, 1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> cols_data = {1, 0, 2, 0, 1, 0, 2, 0};
  std::vector<int64_t> crows_data = {0, 1, 3, 4, 0, 1, 3, 4};
  const int64_t non_zero_num = 8;

  using float16 = phi::dtype::float16;
  std::vector<float16> dense_data_fp16(dense_data.size()),
      non_zero_data_fp16(non_zero_num);
  for (uint64_t i = 0; i < dense_data.size(); i++) {
    dense_data_fp16[i] = static_cast<float16>(dense_data[i]);
  }
  for (int64_t i = 0; i < non_zero_num; i++) {
    non_zero_data_fp16[i] = static_cast<float16>(non_zero_data[i]);
  }
  TestSparseCsrToDense<float16>(dense_dims,
                                dense_data_fp16,
                                non_zero_data_fp16,
                                crows_data,
                                cols_data,
                                non_zero_num);
}

}  // namespace tests
}  // namespace phi
