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

#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/sparse.h"

template <typename T>
void TestNNZ(const std::vector<T>& dense_data, const int correct_nnz,
             const int rows, const int cols) {
  paddle::platform::CUDADeviceContext* context =
      new paddle::platform::CUDADeviceContext(paddle::platform::CUDAPlace());
  context->SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CUDAPlace(), context->stream())
          .get());
  context->PartialInitWithAllocator();
  auto sparse =
      paddle::operators::math::GetSparse<paddle::platform::CUDADeviceContext,
                                         T>(*context);

  paddle::framework::Tensor dense, nnz_tensor;
  auto dense_dims = phi::make_ddim({rows, cols});
  auto nnz_dims = phi::make_ddim({dense_dims[0] + 1});
  dense.mutable_data<T>(dense_dims, paddle::platform::CUDAPlace());
  paddle::framework::TensorFromVector<T>(dense_data, *context, &dense);
  int32_t* nnz_ptr =
      nnz_tensor.mutable_data<int32_t>(nnz_dims, paddle::platform::CUDAPlace());
  sparse.nnz(rows, cols, dense.data<T>(), nnz_ptr, nnz_ptr + 1);
  std::vector<int32_t> nnz_vec(dense_dims[0] + 1);
  paddle::framework::TensorToVector<int32_t>(nnz_tensor, *context, &nnz_vec);
  delete context;
  CHECK_EQ(correct_nnz, nnz_vec[0]);
}

TEST(sparse, nnz) {
  std::vector<float> dense_data = {0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0, 0.0};
  TestNNZ<float>(dense_data, 4, 3, 3);
}

TEST(sparse, nnz_double) {
  std::vector<double> dense_data = {0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0};
  TestNNZ<double>(dense_data, 4, 4, 2);
}

template <typename T>
void TestDenseToSparse(const std::vector<T>& correct_dense_data,
                       const std::vector<int64_t>& correct_rows,
                       const std::vector<int64_t>& correct_cols,
                       const std::vector<T>& correct_values,
                       const int correct_nnz, const int rows, const int cols,
                       const std::string& mode) {
  paddle::platform::CUDADeviceContext* context =
      new paddle::platform::CUDADeviceContext(paddle::platform::CUDAPlace());
  context->SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CUDAPlace(), context->stream())
          .get());
  context->PartialInitWithAllocator();
  // get sparse
  auto sparse =
      paddle::operators::math::GetSparse<paddle::platform::CUDADeviceContext,
                                         T>(*context);

  // create tensor and copy vector to tensor
  paddle::framework::Tensor dense_tensor, rows_tensor, cols_tensor,
      values_tensor, actual_dense_tensor;
  auto dense_dims = phi::make_ddim({rows, cols});
  T* dense_data =
      dense_tensor.mutable_data<T>(dense_dims, paddle::platform::CUDAPlace());
  T* actual_dense_data = actual_dense_tensor.mutable_data<T>(
      dense_dims, paddle::platform::CUDAPlace());
  paddle::framework::TensorFromVector<T>(correct_dense_data, *context,
                                         &dense_tensor);

  auto nnz_dims = phi::make_ddim({correct_nnz});
  auto crows_dims = phi::make_ddim({rows + 1});
  int64_t* rows_data = nullptr;
  if (mode == "COO") {
    rows_data = rows_tensor.mutable_data<int64_t>(
        nnz_dims, paddle::platform::CUDAPlace());
  } else {
    rows_data = rows_tensor.mutable_data<int64_t>(
        crows_dims, paddle::platform::CUDAPlace());
  }
  int64_t* cols_data = cols_tensor.mutable_data<int64_t>(
      nnz_dims, paddle::platform::CUDAPlace());
  T* values_data =
      values_tensor.mutable_data<T>(nnz_dims, paddle::platform::CUDAPlace());

  // test dense_to_sparse
  if (mode == "COO") {
    sparse.DenseToSparseCoo(rows, cols, dense_data, rows_data, cols_data,
                            values_data);
  } else {
    sparse.DenseToSparseCsr(rows, cols, dense_data, rows_data, cols_data,
                            values_data);
  }

  std::vector<int64_t> actual_rows(correct_nnz), actual_crows(rows + 1),
      actual_cols(correct_nnz);
  std::vector<T> actual_values(correct_nnz), actual_dense_vec(rows * cols);
  if (mode == "COO") {
    paddle::framework::TensorToVector<int64_t>(rows_tensor, *context,
                                               &actual_rows);
  } else {
    paddle::framework::TensorToVector<int64_t>(rows_tensor, *context,
                                               &actual_crows);
  }
  paddle::framework::TensorToVector<int64_t>(cols_tensor, *context,
                                             &actual_cols);
  paddle::framework::TensorToVector<T>(values_tensor, *context, &actual_values);

  for (int i = 0; i < correct_nnz; i++) {
    if (mode == "COO") {
      CHECK_EQ(correct_rows[i], actual_rows[i]);
    }
    CHECK_EQ(correct_cols[i], actual_cols[i]);
    CHECK_EQ(correct_values[i], actual_values[i]);
  }
  if (mode == "CSR") {
    for (int i = 0; i < rows + 1; i++) {
      CHECK_EQ(correct_rows[i], actual_crows[i]);
    }
  }

  // test sparse_to_dense
  if (mode == "COO") {
    sparse.SparseCooToDense(rows, cols, correct_nnz, rows_data, cols_data,
                            values_data, actual_dense_data);
  } else {
    sparse.SparseCsrToDense(rows, cols, correct_nnz, rows_data, cols_data,
                            values_data, actual_dense_data);
  }
  paddle::framework::TensorToVector<T>(actual_dense_tensor, *context,
                                       &actual_dense_vec);
  for (uint64_t i = 0; i < correct_dense_data.size(); i++) {
    CHECK_EQ(correct_dense_data[i], actual_dense_vec[i]);
  }

  delete context;
}

TEST(sparse, dense_to_sparse) {
  std::vector<float> dense_data = {0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0, 0.0};
  std::vector<float> values = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> rows = {0, 1, 1, 2};
  std::vector<int64_t> crows = {0, 1, 3, 4};
  std::vector<int64_t> cols = {1, 0, 2, 0};
  TestDenseToSparse<float>(dense_data, rows, cols, values, 4, 3, 3, "COO");
  TestDenseToSparse<float>(dense_data, crows, cols, values, 4, 3, 3, "CSR");
}

TEST(sparse, dense_to_sparse_double) {
  std::vector<double> dense_data = {0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0};
  std::vector<double> values = {1.0, 2.0, 3.0, 3.2};
  std::vector<int64_t> rows = {0, 1, 2, 3};
  std::vector<int64_t> crows = {0, 1, 2, 3, 4};
  std::vector<int64_t> cols = {1, 1, 1, 0};
  TestDenseToSparse<double>(dense_data, rows, cols, values, 4, 4, 2, "COO");
  TestDenseToSparse<double>(dense_data, crows, cols, values, 4, 4, 2, "CSR");
}

TEST(sparse, dense_to_sparse_fp16) {
  using float16 = paddle::platform::float16;
  std::vector<float16> dense_data = {float16(0.0), float16(1.0), float16(0.0),
                                     float16(2.0), float16(0.0), float16(3.0),
                                     float16(3.2), float16(0.0)};
  std::vector<float16> values = {float16(1.0), float16(2.0), float16(3.0),
                                 float16(3.2)};
  std::vector<int64_t> rows = {0, 1, 2, 3};
  std::vector<int64_t> crows = {0, 1, 2, 3, 4};
  std::vector<int64_t> cols = {1, 1, 1, 0};
  TestDenseToSparse<float16>(dense_data, rows, cols, values, 4, 4, 2, "COO");
  TestDenseToSparse<float16>(dense_data, crows, cols, values, 4, 4, 2, "CSR");
}
