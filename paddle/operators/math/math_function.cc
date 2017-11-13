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

#include "paddle/operators/math/math_function.h"
#include "paddle/framework/data_type.h"

namespace paddle {
namespace operators {
namespace math {

template <>
void gemm<platform::CPUPlace, float>(const platform::DeviceContext& context,
                                     const CBLAS_TRANSPOSE transA,
                                     const CBLAS_TRANSPOSE transB, const int M,
                                     const int N, const int K,
                                     const float alpha, const float* A,
                                     const float* B, const float beta,
                                     float* C) {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template <>
void gemm<platform::CPUPlace, double>(const platform::DeviceContext& context,
                                      const CBLAS_TRANSPOSE transA,
                                      const CBLAS_TRANSPOSE transB, const int M,
                                      const int N, const int K,
                                      const double alpha, const double* A,
                                      const double* B, const double beta,
                                      double* C) {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cblas_dgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template <>
void gemm<platform::CPUPlace, float>(const platform::DeviceContext& context,
                                     const bool transA, const bool transB,
                                     const int M, const int N, const int K,
                                     const float alpha, const float* A,
                                     const int lda, const float* B,
                                     const int ldb, const float beta, float* C,
                                     const int ldc) {
  cblas_sgemm(CblasRowMajor, transA == false ? CblasNoTrans : CblasTrans,
              transB == false ? CblasNoTrans : CblasTrans, M, N, K, alpha, A,
              lda, B, ldb, beta, C, ldc);
}

template <>
void gemm<platform::CPUPlace, double>(const platform::DeviceContext& context,
                                      const bool transA, const bool transB,
                                      const int M, const int N, const int K,
                                      const double alpha, const double* A,
                                      const int lda, const double* B,
                                      const int ldb, const double beta,
                                      double* C, const int ldc) {
  cblas_dgemm(CblasRowMajor, transA == false ? CblasNoTrans : CblasTrans,
              transB == false ? CblasNoTrans : CblasTrans, M, N, K, alpha, A,
              lda, B, ldb, beta, C, ldc);
}

template <>
void matmul<platform::CPUPlace, float>(
    const platform::DeviceContext& context, const framework::Tensor& matrix_a,
    bool trans_a, const framework::Tensor& matrix_b, bool trans_b, float alpha,
    framework::Tensor* matrix_out, float beta) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_cpu_place(matrix_a.place()) &&
                     platform::is_cpu_place(matrix_b.place()) &&
                     platform::is_cpu_place(matrix_out->place()),
                 "Matrix must all be in CPUPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  gemm<platform::CPUPlace, float>(
      context, transA, transB, M, N, K, alpha, matrix_a.data<float>(),
      matrix_b.data<float>(), beta, matrix_out->data<float>());
}

template <>
void matmul<platform::CPUPlace, double>(
    const platform::DeviceContext& context, const framework::Tensor& matrix_a,
    bool trans_a, const framework::Tensor& matrix_b, bool trans_b, double alpha,
    framework::Tensor* matrix_out, double beta) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_cpu_place(matrix_a.place()) &&
                     platform::is_cpu_place(matrix_b.place()) &&
                     platform::is_cpu_place(matrix_out->place()),
                 "Matrix must all be in CPUPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  gemm<platform::CPUPlace, double>(
      context, transA, transB, M, N, K, alpha, matrix_a.data<double>(),
      matrix_b.data<double>(), beta, matrix_out->data<double>());
}

#ifdef PADDLE_USE_MKLML
// Use cblas_{s,d}gemm_batched if available: Run with 1 group of size batchSize.
template <>
void batched_gemm<platform::CPUPlace, float>(
    const platform::DeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, const int batchCount, const int strideA, const int strideB) {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  auto a_array = std::vector<const float*>(batchCount);
  auto b_array = std::vector<const float*>(batchCount);
  auto c_array = std::vector<float*>(batchCount);
  for (int k = 0; k < batchCount; ++k) {
    a_array[k] = &A[k * strideA];
    b_array[k] = &B[k * strideB];
    c_array[k] = &C[k * M * N];
  }
  cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &M, &N, &K, &alpha,
                    a_array.data(), &lda, b_array.data(), &ldb, &beta,
                    c_array.data(), &ldc, 1 /* group_count */, &batchCount);
}

template <>
void batched_gemm<platform::CPUPlace, double>(
    const platform::DeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C, const int batchCount, const int strideA, const int strideB) {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  auto a_array = std::vector<const double*>(batchCount);
  auto b_array = std::vector<const double*>(batchCount);
  auto c_array = std::vector<double*>(batchCount);
  for (int k = 0; k < batchCount; ++k) {
    a_array[k] = &A[k * strideA];
    b_array[k] = &B[k * strideB];
    c_array[k] = &C[k * M * N];
  }
  cblas_dgemm_batch(CblasRowMajor, &transA, &transB, &M, &N, &K, &alpha,
                    a_array.data(), &lda, b_array.data(), &ldb, &beta,
                    c_array.data(), &ldc, 1 /* group_count */, &batchCount);
}
#else
// The below is a naive but correct serial implementation that just loops
// over the batch dimension. This is a fallback for when the batched gemm
// functions of Intel MKL are not available. In the future, this computation
// should be parallelized.
template <>
void batched_gemm<platform::CPUPlace, float>(
    const platform::DeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, const int batchCount, const int strideA, const int strideB) {
  for (int k = 0; k < batchCount; ++k) {
    const float* Ak = &A[k * strideA];
    const float* Bk = &B[k * strideB];
    float* Ck = &C[k * M * N];
    gemm<platform::CPUPlace, float>(context, transA, transB, M, N, K, alpha, Ak,
                                    Bk, beta, Ck);
  }
}

template <>
void batched_gemm<platform::CPUPlace, double>(
    const platform::DeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C, const int batchCount, const int strideA, const int strideB) {
  for (int k = 0; k < batchCount; ++k) {
    const double* Ak = &A[k * strideA];
    const double* Bk = &B[k * strideB];
    double* Ck = &C[k * M * N];
    gemm<platform::CPUPlace, double>(context, transA, transB, M, N, K, alpha,
                                     Ak, Bk, beta, Ck);
  }
}
#endif

template <>
void gemv<platform::CPUPlace, float>(const platform::DeviceContext& context,
                                     const bool trans_a, const int M,
                                     const int N, const float alpha,
                                     const float* A, const float* B,
                                     const float beta, float* C) {
  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  cblas_sgemv(CblasRowMajor, transA, M, N, alpha, A, N, B, 1, beta, C, 1);
}

template <>
void gemv<platform::CPUPlace, double>(const platform::DeviceContext& context,
                                      const bool trans_a, const int M,
                                      const int N, const double alpha,
                                      const double* A, const double* B,
                                      const double beta, double* C) {
  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  cblas_dgemv(CblasRowMajor, transA, M, N, alpha, A, N, B, 1, beta, C, 1);
}

template struct SetConstant<platform::CPUPlace, float>;

struct TensorSetConstantCPU {
  TensorSetConstantCPU(framework::Tensor* tensor, float value)
      : tensor_(tensor), value_(value) {}
  template <typename T>
  void operator()() const {
    auto cpu = platform::CPUPlace();
    auto* begin = tensor_->mutable_data<T>(cpu);
    std::fill(begin, begin + tensor_->numel(), static_cast<T>(value_));
  }
  framework::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<platform::CPUPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(framework::ToDataType(tensor->type()),
                           TensorSetConstantCPU(tensor, value));
}

struct TensorSetConstantWithPlace : public boost::static_visitor<void> {
  TensorSetConstantWithPlace(const platform::DeviceContext& context,
                             framework::Tensor* tensor, float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename Place>
  void operator()(Place place) const {
    set_constant_with_place<Place>(context_, tensor_, value_);
  }

  const platform::DeviceContext& context_;
  framework::Tensor* tensor_;
  float value_;
};

void set_constant(const platform::DeviceContext& context,
                  framework::Tensor* tensor, float value) {
  TensorSetConstantWithPlace func(context, tensor, value);
#ifdef PADDLE_WITH_CUDA
  tensor->place().apply_visitor(func);
#else
  func(platform::CPUPlace());
#endif
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
