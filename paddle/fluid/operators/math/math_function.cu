/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#define EIGEN_USE_GPU
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/math_function_impl.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

using float16 = paddle::platform::float16;

template <>
void matmul<platform::CUDADeviceContext, float16>(
    const platform::CUDADeviceContext& context,
    const framework::Tensor& matrix_a, bool trans_a,
    const framework::Tensor& matrix_b, bool trans_b, float16 alpha,
    framework::Tensor* matrix_out, float16 beta) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_gpu_place(matrix_a.place()) &&
                     platform::is_gpu_place(matrix_b.place()) &&
                     platform::is_gpu_place(matrix_out->place()),
                 "Matrix must all be in CUDAPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  Blas<platform::CUDADeviceContext>(context).GEMM(
      transA, transB, M, N, K, alpha, matrix_a.data<float16>(),
      matrix_b.data<float16>(), beta, matrix_out->data<float16>());
}

template <>
void matmul<platform::CUDADeviceContext, float>(
    const platform::CUDADeviceContext& context,
    const framework::Tensor& matrix_a, bool trans_a,
    const framework::Tensor& matrix_b, bool trans_b, float alpha,
    framework::Tensor* matrix_out, float beta) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_gpu_place(matrix_a.place()) &&
                     platform::is_gpu_place(matrix_b.place()) &&
                     platform::is_gpu_place(matrix_out->place()),
                 "Matrix must all be in CUDAPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  Blas<platform::CUDADeviceContext>(context).GEMM(
      transA, transB, M, N, K, alpha, matrix_a.data<float>(),
      matrix_b.data<float>(), beta, matrix_out->data<float>());
}

template <>
void matmul<platform::CUDADeviceContext, double>(
    const platform::CUDADeviceContext& context,
    const framework::Tensor& matrix_a, bool trans_a,
    const framework::Tensor& matrix_b, bool trans_b, double alpha,
    framework::Tensor* matrix_out, double beta) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_gpu_place(matrix_a.place()) &&
                     platform::is_gpu_place(matrix_b.place()) &&
                     platform::is_gpu_place(matrix_out->place()),
                 "Matrix must all be in CUDAPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  Blas<platform::CUDADeviceContext>(context).GEMM(
      transA, transB, M, N, K, alpha, matrix_a.data<double>(),
      matrix_b.data<double>(), beta, matrix_out->data<double>());
}

template <>
void batched_gemm<platform::CUDADeviceContext, float16>(
    const platform::CUDADeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const float16 alpha, const float16* A, const float16* B, const float16 beta,
    float16* C, const int batchCount, const int64_t strideA,
    const int64_t strideB) {
#if CUDA_VERSION >= 8000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

  const half h_alpha = static_cast<const half>(alpha);
  const half h_beta = static_cast<const half>(beta);
  const half* h_A = reinterpret_cast<const half*>(A);
  const half* h_B = reinterpret_cast<const half*>(B);
  half* h_C = reinterpret_cast<half*>(C);

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(context.GetComputeCapability(), 53,
                    "cublas Hgemm requires GPU compute capability >= 53");

  PADDLE_ENFORCE(platform::dynload::cublasHgemmStridedBatched(
      context.cublas_handle(), cuTransB, cuTransA, N, M, K, &h_alpha, h_B, ldb,
      strideB, h_A, lda, strideA, &h_beta, h_C, ldc, strideC, batchCount));
#else
  PADDLE_ENFORCE(false, "HgemmStridedBatched is not supported on cuda <= 7.5");
#endif
}

template <>
void batched_gemm<platform::CUDADeviceContext, float>(
    const platform::CUDADeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, const int batchCount, const int64_t strideA,
    const int64_t strideB) {
#if CUDA_VERSION >= 8000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

  PADDLE_ENFORCE(platform::dynload::cublasSgemmStridedBatched(
      context.cublas_handle(), cuTransB, cuTransA, N, M, K, &alpha, B, ldb,
      strideB, A, lda, strideA, &beta, C, ldc, strideC, batchCount));
#else
  PADDLE_ENFORCE(false, "SgemmStridedBatched is not supported on cuda <= 7.5");
#endif
}

template <>
void batched_gemm<platform::CUDADeviceContext, double>(
    const platform::CUDADeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C, const int batchCount, const int64_t strideA,
    const int64_t strideB) {
#if CUDA_VERSION >= 8000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

  PADDLE_ENFORCE(platform::dynload::cublasDgemmStridedBatched(
      context.cublas_handle(), cuTransB, cuTransA, N, M, K, &alpha, B, ldb,
      strideB, A, lda, strideA, &beta, C, ldc, strideC, batchCount));
#else
  PADDLE_ENFORCE(false, "DgemmStridedBatched is not supported on cuda <= 7.5");
#endif
}

template <>
void gemv<platform::CUDADeviceContext, float>(
    const platform::CUDADeviceContext& context, const bool trans_a, const int M,
    const int N, const float alpha, const float* A, const float* B,
    const float beta, float* C) {
  cublasOperation_t cuTransA = (trans_a == false) ? CUBLAS_OP_T : CUBLAS_OP_N;

  PADDLE_ENFORCE(platform::dynload::cublasSgemv(context.cublas_handle(),
                                                cuTransA, N, M, &alpha, A, N, B,
                                                1, &beta, C, 1));
}

template <>
void gemv<platform::CUDADeviceContext, double>(
    const platform::CUDADeviceContext& context, const bool trans_a, const int M,
    const int N, const double alpha, const double* A, const double* B,
    const double beta, double* C) {
  cublasOperation_t cuTransA = (trans_a == false) ? CUBLAS_OP_T : CUBLAS_OP_N;
  PADDLE_ENFORCE(platform::dynload::cublasDgemv(context.cublas_handle(),
                                                cuTransA, N, M, &alpha, A, N, B,
                                                1, &beta, C, 1));
}

template <>
void axpy<platform::CUDADeviceContext, float>(
    const platform::CUDADeviceContext& context, const int n, const float alpha,
    const float* x, float* y) {
  PADDLE_ENFORCE(platform::dynload::cublasSaxpy(context.cublas_handle(), n,
                                                &alpha, x, 1, y, 1));
}

template <>
void axpy<platform::CUDADeviceContext, double>(
    const platform::CUDADeviceContext& context, const int n, const double alpha,
    const double* x, double* y) {
  PADDLE_ENFORCE(platform::dynload::cublasDaxpy(context.cublas_handle(), n,
                                                &alpha, x, 1, y, 1));
}

template struct SetConstant<platform::CUDADeviceContext, platform::float16>;
template struct SetConstant<platform::CUDADeviceContext, float>;
template struct SetConstant<platform::CUDADeviceContext, double>;
template struct SetConstant<platform::CUDADeviceContext, int>;
template struct SetConstant<platform::CUDADeviceContext, int64_t>;
template struct SetConstant<platform::CUDADeviceContext, bool>;

#define DEFINE_GPU_TRANS(RANK)                                         \
  template struct Transpose<platform::CUDADeviceContext, float, RANK>; \
  template struct Transpose<platform::CUDADeviceContext, double, RANK>;

DEFINE_GPU_TRANS(1);
DEFINE_GPU_TRANS(2);
DEFINE_GPU_TRANS(3);
DEFINE_GPU_TRANS(4);
DEFINE_GPU_TRANS(5);
DEFINE_GPU_TRANS(6);

struct TensorSetConstantGPU {
  TensorSetConstantGPU(const platform::DeviceContext& context,
                       framework::Tensor* tensor, float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename T>
  void operator()() const {
    SetConstant<platform::CUDADeviceContext, T> functor;
    functor(reinterpret_cast<const platform::CUDADeviceContext&>(context_),
            tensor_, static_cast<T>(value_));
  }

  const platform::DeviceContext& context_;
  framework::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<platform::CUDAPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(framework::ToDataType(tensor->type()),
                           TensorSetConstantGPU(context, tensor, value));
}

template <typename T>
__global__ void RowwiseAddKernel(const T* a, const T* b, T* c, int width,
                                 int num) {
  T tmp = 1.0 / width;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    int h = i * tmp;
    int w = i - h * width;
    c[i] = a[i] + b[w];
  }
}

template <typename T>
struct RowwiseAdd<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& vector, framework::Tensor* output) {
    auto in_dims = input.dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_ENFORCE_EQ(vector.numel(), size);
    PADDLE_ENFORCE_EQ(output->dims(), in_dims);
    int blocks = 512;
    int grids = (input.numel() + blocks - 1) / blocks;
    RowwiseAddKernel<T><<<grids, blocks, 0, context.stream()>>>(
        input.data<T>(), vector.data<T>(), output->data<T>(),
        static_cast<int>(in_dims[1]), static_cast<int>(input.numel()));
  }
};

template struct RowwiseAdd<platform::CUDADeviceContext, float>;
template struct RowwiseAdd<platform::CUDADeviceContext, double>;
template struct ColwiseSum<platform::CUDADeviceContext, float>;
template struct ColwiseSum<platform::CUDADeviceContext, int>;
template struct ColwiseSum<platform::CUDADeviceContext, int64_t>;
// template struct ColwiseSum<platform::CUDADeviceContext, double>;
// The ColwiseSum<platform::CUDADeviceContext, double> failed in debug mode,
// and only failed for this case. So reimplemented it.
template <>
void ColwiseSum<platform::CUDADeviceContext, double>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor& input,
    framework::Tensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(), size);
  framework::Tensor one;
  one.mutable_data<double>({in_dims[0]}, context.GetPlace());
  SetConstant<platform::CUDADeviceContext, double> set;
  set(context, &one, static_cast<double>(1.0));
  gemv<platform::CUDADeviceContext, double>(
      context, true, static_cast<int>(in_dims[0]), static_cast<int>(in_dims[1]),
      1.0, input.data<double>(), one.data<double>(), 0.0,
      vector->data<double>());
}

template struct RowwiseSum<platform::CUDADeviceContext, float>;
// template struct RowwiseSum<platform::CUDADeviceContext, double>;
// TODO(zcd): Following ColwiseSum format, need to confirm.
// The RowwiseSum<platform::CUDADeviceContext, double> failed in debug mode,
// and only failed for this case. So reimplemented it.
template <>
void RowwiseSum<platform::CUDADeviceContext, double>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor& input,
    framework::Tensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(), in_dims[0]);
  framework::Tensor one;
  one.mutable_data<double>({size}, context.GetPlace());
  SetConstant<platform::CUDADeviceContext, double> set;
  set(context, &one, static_cast<double>(1.0));
  gemv<platform::CUDADeviceContext, double>(
      context, true, static_cast<int>(in_dims[1]), static_cast<int>(in_dims[0]),
      1.0, one.data<double>(), input.data<double>(), 0.0,
      vector->data<double>());
}

template struct RowwiseMean<platform::CUDADeviceContext, float>;
template struct RowwiseMean<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
