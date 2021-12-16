//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/dynload/cusparse.h"

#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
cudaDataType_t GetGpuDataType() {
  if (std::is_same<T, float>::value) {
    return CUDA_R_32F;
  } else if (std::is_same<T, double>::value) {
    return CUDA_R_64F;
  }
}

template <>
template <typename T>
void Sparse<platform::CUDADeviceContext>::nnz(const int M, const int N,
                                              const T* dense, int* nnz,
                                              int* nnzPerRowColumn) const {}

template <>
template <>
void Sparse<platform::CUDADeviceContext>::nnz(const int M, const int N,
                                              const float* dense, int* nnz,
                                              int* nnzPerRowColumn) const {
  cusparseMatDescr_t descr = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cusparseCreateMatDescr(&descr));
  PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::cusparseSetMatType(
      descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::cusparseSetMatIndexBase(
      descr, CUSPARSE_INDEX_BASE_ZERO));

  context_.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::cusparseSnnz(
        handle, CUSPARSE_DIRECTION_ROW, M, N, descr, dense, M, nnzPerRowColumn,
        nnz));
  });
}

template <>
template <>
void Sparse<platform::CUDADeviceContext>::nnz(const int M, const int N,
                                              const double* dense, int* nnz,
                                              int* nnzPerRowColumn) const {
  cusparseMatDescr_t descr = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cusparseCreateMatDescr(&descr));
  PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::cusparseSetMatType(
      descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::cusparseSetMatIndexBase(
      descr, CUSPARSE_INDEX_BASE_ZERO));

  context_.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::cusparseDnnz(
        handle, CUSPARSE_DIRECTION_ROW, M, N, descr, dense, M, nnzPerRowColumn,
        nnz));
  });
}

template <>
template <typename T>
void Sparse<platform::CUDADeviceContext>::DenseToSparseCoo(
    const int M, const int N, const T* dense, int64_t* rows, int64_t* cols,
    T* values) const {
  cusparseSpMatDescr_t matA, matB;

  cudaDataType_t dtype = GetGpuDataType<T>();

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateDnMat(
      &matA, M, N, N, dense, dtype, CUSPARSE_ORDER_ROW));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateCoo(
      &matB, M, N, 0, NULL, NULL, NULL, CUSPARSE_INDEX_64I,
      CUSPARSE_INDEX_BASE_ZERO, dtype));

  size_t buffer_size = 0;
  context_.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cusparseDenseToSparse_bufferSize(
            handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            &buffer_size));
  });
  framework::Tensor buffer;
  T* buffer_data = buffer.mutable_data<T>({buffer_size}, context_.GetPlace());

  context_.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cusparseDenseToSparse_analysis(
            handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            buffer_data));
  });

  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cusparseCooSetPointers(matB, rows, cols, values));
  context_.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseDenseToSparse_convert(
        handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer_data));
  });
}

template <>
template <typename T>
void Sparse<platform::CUDADeviceContext>::DenseToSparseCsr(
    const int M, const int N, const T* dense, int64_t* crows, int64_t* cols,
    T* values) const {
  cusparseSpMatDescr_t matB;
  cusparseDnMatDescr_t matA;

  cudaDataType_t dtype = GetGpuDataType<T>();

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateDnMat(
      &matA, M, N, N, const_cast<void*>(reinterpret_cast<const void*>(dense)),
      dtype, CUSPARSE_ORDER_ROW));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateCsr(
      &matB, M, N, 0, crows, nullptr, nullptr, CUSPARSE_INDEX_64I,
      CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, dtype));

  size_t buffer_size = 0;
  context_.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cusparseDenseToSparse_bufferSize(
            handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            &buffer_size));
  });
  framework::Tensor buffer;
  float* buffer_data = buffer.mutable_data<float>(
      {static_cast<int64_t>(buffer_size)}, context_.GetPlace());

  context_.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cusparseDenseToSparse_analysis(
            handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            buffer_data));
  });

  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cusparseCsrSetPointers(matB, crows, cols, values));
  context_.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseDenseToSparse_convert(
        handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer_data));
  });
}

template <>
template <typename T>
void Sparse<platform::CUDADeviceContext>::SparseCsrToDense(
    const int M, const int N, const int nnz, const int64_t* crows,
    const int64_t* cols, const T* values, T* dense) const {
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB;

  cudaDataType_t dtype = GetGpuDataType<T>();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateCsr(
      &matA, M, N, nnz, crows, cols, values, CUSPARSE_INDEX_64I,
      CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, dtype));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateDnMat(
      &matB, M, N, N, const_cast<void*>(reinterpret_cast<const void*>(dense)),
      dtype, CUSPARSE_ORDER_ROW));

  size_t buffer_size = 0;
  context_.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cusparseSparseToDense_bufferSize(
            handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
            &buffer_size));
  });
  framework::Tensor buffer;
  float* buffer_data = buffer.mutable_data<float>(
      {static_cast<int64_t>(buffer_size)}, context_.GetPlace());

  context_.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseSparseToDense(
        handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer_data));
  });
}
}  // namespace math
}  // namespace operators
}  // namespace paddle
