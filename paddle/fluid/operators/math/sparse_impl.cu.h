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

#include "paddle/fluid/platform/dynload/cusparse.h"
#include "paddle/pten/kernels/funcs/math_function.h"

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
  } else if (std::is_same<T, platform::float16>::value) {
    return CUDA_R_16F;
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

template <typename T>
inline void DenseToSparse(const platform::CUDADeviceContext& context,
                          const int M, const int N, const T* dense,
                          int64_t* rows, int64_t* cols, T* values,
                          const cusparseFormat_t format) {
  cusparseSpMatDescr_t matB;
  cusparseDnMatDescr_t matA;

  cudaDataType_t dtype = GetGpuDataType<T>();

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateDnMat(
      &matA, M, N, N, const_cast<void*>(reinterpret_cast<const void*>(dense)),
      dtype, CUSPARSE_ORDER_ROW));

  if (format == CUSPARSE_FORMAT_COO) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateCoo(
        &matB, M, N, 0, nullptr, nullptr, nullptr, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, dtype));
  } else if (format == CUSPARSE_FORMAT_CSR) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateCsr(
        &matB, M, N, 0, rows, nullptr, nullptr, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, dtype));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "the sparse format [%s] is not supported", format));
  }

  size_t buffer_size = 0;
  context.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cusparseDenseToSparse_bufferSize(
            handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            &buffer_size));
  });
  framework::Tensor buffer;
  float* buffer_data = buffer.mutable_data<float>(
      {static_cast<int64_t>(buffer_size)}, context.GetPlace());

  context.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cusparseDenseToSparse_analysis(
            handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            buffer_data));
  });

  if (format == CUSPARSE_FORMAT_COO) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCooSetPointers(
        matB, rows, cols, reinterpret_cast<void*>(values)));
  } else if (format == CUSPARSE_FORMAT_CSR) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCsrSetPointers(
        matB, rows, cols, reinterpret_cast<void*>(values)));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "the sparse format [%s] is not supported", format));
  }
  context.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseDenseToSparse_convert(
        handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer_data));
  });
}
template <>
template <typename T>
void Sparse<platform::CUDADeviceContext>::DenseToSparseCoo(
    const int M, const int N, const T* dense, int64_t* rows, int64_t* cols,
    T* values) const {
  DenseToSparse<T>(context_, M, N, dense, rows, cols, values,
                   CUSPARSE_FORMAT_COO);
}

template <>
template <typename T>
void Sparse<platform::CUDADeviceContext>::DenseToSparseCsr(
    const int M, const int N, const T* dense, int64_t* crows, int64_t* cols,
    T* values) const {
  DenseToSparse<T>(context_, M, N, dense, crows, cols, values,
                   CUSPARSE_FORMAT_CSR);
}

template <typename T>
void SparseToDense(const platform::CUDADeviceContext& context, const int64_t M,
                   const int64_t N, const int64_t nnz, const int64_t* rows,
                   const int64_t* cols, const T* values, T* dense,
                   const cusparseFormat_t format) {
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB;

  cudaDataType_t dtype = GetGpuDataType<T>();
  if (format == CUSPARSE_FORMAT_COO) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateCoo(
        &matA, M, N, nnz,
        const_cast<void*>(reinterpret_cast<const void*>(rows)),
        const_cast<void*>(reinterpret_cast<const void*>(cols)),
        const_cast<void*>(reinterpret_cast<const void*>(values)),
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, dtype));
  } else if (format == CUSPARSE_FORMAT_CSR) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateCsr(
        &matA, M, N, nnz,
        const_cast<void*>(reinterpret_cast<const void*>(rows)),
        const_cast<void*>(reinterpret_cast<const void*>(cols)),
        const_cast<void*>(reinterpret_cast<const void*>(values)),
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO,
        dtype));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "the sparse format [%s] is not supported", format));
  }

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseCreateDnMat(
      &matB, M, N, N, reinterpret_cast<void*>(dense), dtype,
      CUSPARSE_ORDER_ROW));

  size_t buffer_size = 0;
  context.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cusparseSparseToDense_bufferSize(
            handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
            &buffer_size));
  });
  framework::Tensor buffer;
  float* buffer_data = buffer.mutable_data<float>(
      {static_cast<int64_t>(buffer_size)}, context.GetPlace());

  context.CusparseCall([&](cusparseHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusparseSparseToDense(
        handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer_data));
  });
}

template <>
template <typename T>
void Sparse<platform::CUDADeviceContext>::SparseCooToDense(
    const int64_t M, const int64_t N, const int64_t nnz, const int64_t* rows,
    const int64_t* cols, const T* values, T* dense) const {
  SparseToDense<T>(context_, M, N, nnz, rows, cols, values, dense,
                   CUSPARSE_FORMAT_COO);
}

template <>
template <typename T>
void Sparse<platform::CUDADeviceContext>::SparseCsrToDense(
    const int64_t M, const int64_t N, const int64_t nnz, const int64_t* crows,
    const int64_t* cols, const T* values, T* dense) const {
  SparseToDense<T>(context_, M, N, nnz, crows, cols, values, dense,
                   CUSPARSE_FORMAT_CSR);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
