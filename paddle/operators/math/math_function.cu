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

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <>
void gemm<platform::GPUPlace, float>(const CBLAS_TRANSPOSE transA,
                                     const CBLAS_TRANSPOSE transB, const int M,
                                     const int N, const int K,
                                     const float alpha, const float* A,
                                     const float* B, const float beta, float* C,
                                     platform::DeviceContext* context) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE(platform::dynload::cublasSgemm(
      reinterpret_cast<platform::CUDADeviceContext*>(context)->cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void gemm<platform::GPUPlace, double>(const CBLAS_TRANSPOSE transA,
                                      const CBLAS_TRANSPOSE transB, const int M,
                                      const int N, const int K,
                                      const double alpha, const double* A,
                                      const double* B, const double beta,
                                      double* C,
                                      platform::DeviceContext* context) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  PADDLE_ENFORCE(platform::dynload::cublasDgemm(
      reinterpret_cast<platform::CUDADeviceContext*>(context)->cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void matmul<platform::GPUPlace, float>(const framework::Tensor& matrix_a,
                                       bool trans_a,
                                       const framework::Tensor& matrix_b,
                                       bool trans_b, float alpha,
                                       framework::Tensor* matrix_out,
                                       float beta,
                                       platform::DeviceContext* context) {
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
      transA, transB, M, N, K, alpha, matrix_a.data<float>(),
      matrix_b.data<float>(), beta, matrix_out->data<float>(), context);
}

template <>
void matmul<platform::GPUPlace, double>(const framework::Tensor& matrix_a,
                                        bool trans_a,
                                        const framework::Tensor& matrix_b,
                                        bool trans_b, double alpha,
                                        framework::Tensor* matrix_out,
                                        double beta,
                                        platform::DeviceContext* context) {
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
      transA, transB, M, N, K, alpha, matrix_a.data<double>(),
      matrix_b.data<double>(), beta, matrix_out->data<double>(), context);
}

template <>
void Set<typename GPUPlace, typename float>(const int n, const float alpha,
                                            float* output,
                                            platform::DeviceContext* context) {
  auto* cuda_context = reinterpret_cast<platform::CUDADeviceContext*>(context);
  framework::EigenVector::Type<T> out(output, n);
  out.device(*(cuda_context->eigen_device())) = t.constant(T(alpha));
}

template <typename T>
__global__ void UniformShift(const int n, const T min, const T max, T* x) {
  float scale = max - min;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    x[i] = x[i] * scale + min;
  }
}

template <>
void RandUniform<platform::GPUPlace, float>(const int n, const float min,
                                            const float max, float* output,
                                            platform::DeviceContext* context) {
  auto* cuda_context = reinterpret_cast<platform::CUDADeviceContext*>(context);
  PADDLE_ENFORCE(
      curandGenerateUniform(cuda_context->curand_generator(), output, n));
  int block = 512;
  int grid = (n + block - 1) / block;
  UniformShift<float><<<grid, block, 0, cuda_context->stream()>>>(n, min, max,
                                                                  output);
}

template <typename T>
int HandleOddLengthRandGaussian(const int n, const T mean, const T std,
                                T* output, CUDADeviceContext* context) {
  if (n % 2 == 1) {
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(mean, std);
    const T random_value = distribution(generator);
    Set<T, platform::GPUPlace>(1, random_value, output + (n - 1), context);
    return n - 1;
  }
  return n;
}

template <>
void RandGaussian<platform::GPUPlace, float>(const int n, const float mean,
                                             const float std, float* output,
                                             platform::DeviceContext* context) {
  auto* cuda_context = reinterpret_cast<platform::CUDADeviceContext*>(context);

  const int even_n =
      HandleOddLengthRandGaussian<float>(n, mean, std, output, cuda_context);
  PADDLE_ENFORCE(curandGenerateNormal(cuda_context->curand_generator(), output,
                                      even_n, mean, std));
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
