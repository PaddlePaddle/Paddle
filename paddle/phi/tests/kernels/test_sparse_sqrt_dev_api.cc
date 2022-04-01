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

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/sparse_sqrt_kernel.h"

namespace phi {
namespace tests {

template <typename T1, typename T2>
std::vector<T2> cast(const std::vector<T1>& in) {
  std::vector<T2> out(in.size());
  for (uint64_t i = 0; i < in.size(); i++) {
    out[i] = static_cast<T2>(in[i]);
  }
  return out;
}

template <typename T>
void TestSqrtCsrBase(const DDim& dense_dims,
                     const std::vector<T>& non_zero_data,
                     const std::vector<int64_t>& crows_data,
                     const std::vector<int64_t>& cols_data,
                     const int64_t non_zero_num,
                     const std::vector<T>& correct_out_values,
                     const float eps = 1e-3) {
  int64_t batchs = 1;
  int64_t rows = dense_dims[0];
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
  phi::SparseCsrTensor x(crows, cols, values, dense_dims);

  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  auto out = sparse::Sqrt<T>(dev_ctx_cpu, x);
  //  auto cpu_sparse_out = sparse::SparseCsrToCoo<T>(dev_ctx_cpu, csr);
  auto f_verify = [&](const T* real_data, const std::vector<T>& correct_data) {
    for (uint64_t i = 0; i < correct_data.size(); i++) {
      float tmp = std::fabs(static_cast<float>(correct_data[i] - real_data[i]));
      ASSERT_LT(tmp, eps);
    }
  };

  f_verify(out.non_zero_elements().template data<T>(), correct_out_values);
}

void TestSqrtCsr(const DDim& dense_dims,
                 const std::vector<float>& non_zero_data,
                 const std::vector<int64_t>& crows_data,
                 const std::vector<int64_t>& cols_data,
                 const int64_t non_zero_num,
                 const std::vector<float>& correct_out_values,
                 const float eps = 1e-3) {
  TestSqrtCsrBase<float>(dense_dims,
                         non_zero_data,
                         crows_data,
                         cols_data,
                         non_zero_num,
                         correct_out_values,
                         eps);
  TestSqrtCsrBase<double>(dense_dims,
                          cast<float, double>(non_zero_data),
                          crows_data,
                          cols_data,
                          non_zero_num,
                          cast<float, double>(correct_out_values),
                          eps);
}

TEST(DEV_API, sparse_sqrtcsr) {
  DDim x_dims = {3, 3};

  const int non_zero_num = 6;
  std::vector<int64_t> crows = {0, 2, 3, 6};
  std::vector<int64_t> cols = {0, 2, 2, 0, 1, 2};
  std::vector<float> values = {1, 2, 3, 4, 5, 6};

  std::vector<float> out_values(values);
  std::for_each(std::begin(out_values), std::end(out_values), [&](float& a) {
    a = sqrt(a);
  });

  TestSqrtCsr(x_dims, values, crows, cols, non_zero_num, out_values);
}

TEST(DEV_API, sparse_sqrtcsr_batch) {
  DDim x_dims = {2, 2, 4};
  const int non_zero_num = 11;
  std::vector<int64_t> crows = {0, 1, 3, 4, 6, 0, 1, 2, 4, 5};
  std::vector<int64_t> cols = {1, 0, 3, 2, 1, 3, 1, 0, 3, 2, 1};
  std::vector<float> values = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5};

  std::vector<float> out_values(values);
  std::for_each(std::begin(out_values), std::end(out_values), [&](float& a) {
    a = sqrt(a);
  });

  TestSqrtCsr(x_dims, values, crows, cols, non_zero_num, out_values);
}

template <typename T>
void TestSqrtCooBase(const DDim& dense_dims,
                     const std::vector<T>& non_zero_data,
                     const std::vector<int64_t>& non_zero_indices,
                     const int64_t non_zero_num,
                     const std::vector<T>& correct_out_values,
                     const float eps = 1e-3) {
//  int64_t batchs = 1;
//  int64_t rows = dense_dims[0];
//  if (dense_dims.size() == 3) {
//    batchs = dense_dims[0];
//    rows = dense_dims[1];
//  }
  phi::CPUPlace place;
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  DenseTensorMeta indices_meta(
      DataType::INT64,
      {static_cast<int64_t>(dense_dims.size()), non_zero_num},
      DataLayout::NCHW);
  DenseTensorMeta values_meta(
      paddle::experimental::CppTypeToDataType<T>::Type(),
      {non_zero_num},
      DataLayout::NCHW);

  DenseTensor indices(alloc.get(), indices_meta);
  DenseTensor values(alloc.get(), values_meta);

  memcpy(indices.mutable_data<int64_t>(place),
         non_zero_indices.data(),
         non_zero_indices.size() * sizeof(int64_t));
  memcpy(values.mutable_data<T>(place),
         non_zero_data.data(),
         non_zero_data.size() * sizeof(T));

  phi::SparseCooTensor x(indices, values, dense_dims);

  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  auto out = sparse::Sqrt<T>(dev_ctx_cpu, x);

  auto f_verify = [&](const T* real_data, const std::vector<T>& correct_data) {
    for (uint64_t i = 0; i < correct_data.size(); i++) {
      float tmp = std::fabs(static_cast<float>(correct_data[i] - real_data[i]));
      ASSERT_LT(tmp, eps);
    }
  };

  f_verify(out.non_zero_elements().template data<T>(), correct_out_values);
}

void TestSqrtCoo(const DDim& dense_dims,
                 const std::vector<float>& non_zero_data,
                 const std::vector<int64_t>& non_zero_indices,
                 const int64_t non_zero_num,
                 const std::vector<float>& correct_out_values,
                 const float eps = 1e-3) {
  TestSqrtCooBase<float>(dense_dims,
                         non_zero_data,
                         non_zero_indices,
                         non_zero_num,
                         correct_out_values,
                         eps);
  TestSqrtCooBase<double>(dense_dims,
                          cast<float, double>(non_zero_data),
                          non_zero_indices,
                          non_zero_num,
                          cast<float, double>(correct_out_values),
                          eps);
}

TEST(DEV_API, sparse_sqrtcoo) {
  DDim x_dims = {3, 3};
  //  array([[1, 0, 2],
  //          [0, 0, 3],
  //          [4, 5, 6]])
  const int non_zero_num = 6;
  std::vector<int64_t> indices = {0, 0, 1, 2, 2, 2, 0, 2, 2, 0, 1, 2};
  std::vector<float> values = {1, 2, 3, 4, 5, 6};

  std::vector<float> out_values(values);
  std::for_each(std::begin(out_values), std::end(out_values), [&](float& a) {
    a = sqrt(a);
  });

  TestSqrtCoo(x_dims, values, indices, non_zero_num, out_values);
}

}  // namespace tests
}  // namespace phi
