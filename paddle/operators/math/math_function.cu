/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/data_type.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <>
void gemm<platform::GPUPlace, float>(const platform::DeviceContext& context,
                                     const CBLAS_TRANSPOSE transA,
                                     const CBLAS_TRANSPOSE transB, const int M,
                                     const int N, const int K,
                                     const float alpha, const float* A,
                                     const float* B, const float beta,
                                     float* C) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE(platform::dynload::cublasSgemm(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void gemm<platform::GPUPlace, double>(const platform::DeviceContext& context,
                                      const CBLAS_TRANSPOSE transA,
                                      const CBLAS_TRANSPOSE transB, const int M,
                                      const int N, const int K,
                                      const double alpha, const double* A,
                                      const double* B, const double beta,
                                      double* C) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  PADDLE_ENFORCE(platform::dynload::cublasDgemm(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void gemm<platform::GPUPlace, float>(const platform::DeviceContext& context,
                                     const bool transA, const bool transB,
                                     const int M, const int N, const int K,
                                     const float alpha, const float* A,
                                     const int lda, const float* B,
                                     const int ldb, const float beta, float* C,
                                     const int ldc) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA == false ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = transB == false ? CUBLAS_OP_N : CUBLAS_OP_T;
  PADDLE_ENFORCE(platform::dynload::cublasSgemm(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
}

template <>
void gemm<platform::GPUPlace, double>(const platform::DeviceContext& context,
                                      const bool transA, const bool transB,
                                      const int M, const int N, const int K,
                                      const double alpha, const double* A,
                                      const int lda, const double* B,
                                      const int ldb, const double beta,
                                      double* C, const int ldc) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA == false ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = transB == false ? CUBLAS_OP_N : CUBLAS_OP_T;
  PADDLE_ENFORCE(platform::dynload::cublasDgemm(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
}

template <>
void matmul<platform::GPUPlace, float>(
    const platform::DeviceContext& context, const framework::Tensor& matrix_a,
    bool trans_a, const framework::Tensor& matrix_b, bool trans_b, float alpha,
    framework::Tensor* matrix_out, float beta) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_gpu_place(matrix_a.place()) &&
                     platform::is_gpu_place(matrix_b.place()) &&
                     platform::is_gpu_place(matrix_out->place()),
                 "Matrix must all be in GPUPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  gemm<platform::GPUPlace, float>(
      context, transA, transB, M, N, K, alpha, matrix_a.data<float>(),
      matrix_b.data<float>(), beta, matrix_out->data<float>());
}

template <>
void matmul<platform::GPUPlace, double>(
    const platform::DeviceContext& context, const framework::Tensor& matrix_a,
    bool trans_a, const framework::Tensor& matrix_b, bool trans_b, double alpha,
    framework::Tensor* matrix_out, double beta) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_gpu_place(matrix_a.place()) &&
                     platform::is_gpu_place(matrix_b.place()) &&
                     platform::is_gpu_place(matrix_out->place()),
                 "Matrix must all be in GPUPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  gemm<platform::GPUPlace, double>(
      context, transA, transB, M, N, K, alpha, matrix_a.data<double>(),
      matrix_b.data<double>(), beta, matrix_out->data<double>());
}

template <>
void batched_gemm<platform::GPUPlace, float>(
    const platform::DeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, const int batchCount, const int strideA, const int strideB) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int strideC = M * N;

  PADDLE_ENFORCE(platform::dynload::cublasSgemmStridedBatched(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, strideB, A, lda, strideA,
      &beta, C, ldc, strideC, batchCount));
}

template <>
void batched_gemm<platform::GPUPlace, double>(
    const platform::DeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C, const int batchCount, const int strideA, const int strideB) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int strideC = M * N;

  PADDLE_ENFORCE(platform::dynload::cublasDgemmStridedBatched(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, strideB, A, lda, strideA,
      &beta, C, ldc, strideC, batchCount));
}

template <>
void gemv<platform::GPUPlace, float>(const platform::DeviceContext& context,
                                     const bool trans_a, const int M,
                                     const int N, const float alpha,
                                     const float* A, const float* B,
                                     const float beta, float* C) {
  cublasOperation_t cuTransA = (trans_a == false) ? CUBLAS_OP_T : CUBLAS_OP_N;

  PADDLE_ENFORCE(platform::dynload::cublasSgemv(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransA, N, M, &alpha, A, N, B, 1, &beta, C, 1));
}

template <>
void gemv<platform::GPUPlace, double>(const platform::DeviceContext& context,
                                      const bool trans_a, const int M,
                                      const int N, const double alpha,
                                      const double* A, const double* B,
                                      const double beta, double* C) {
  cublasOperation_t cuTransA = (trans_a == false) ? CUBLAS_OP_T : CUBLAS_OP_N;
  PADDLE_ENFORCE(platform::dynload::cublasDgemv(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransA, N, M, &alpha, A, N, B, 1, &beta, C, 1));
}

template struct SetConstant<platform::GPUPlace, float>;

struct TensorSetConstantGPU {
  TensorSetConstantGPU(const platform::DeviceContext& context,
                    framework::Tensor* tensor, float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename T>
  void operator()() const {
    SetConstant<platform::GPUPlace, T> functor;
    functor(context_, tensor_, static_cast<T>(value_));
  }

  const platform::DeviceContext& context_;
  framework::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<platform::GPUPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(framework::ToDataType(tensor->type()),
                           TensorSetConstantGPU(context, tensor, value));
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
