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

#pragma once
#ifdef PADDLE_WITH_MKLML
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <mkl_vml_functions.h>
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#include <lapacke.h>
#endif

#ifndef LAPACK_FOUND
extern "C" {
#include <cblas.h>  // NOLINT
int LAPACKE_sgetrf(int matrix_layout, int m, int n, float* a, int lda,
                   int* ipiv);
int LAPACKE_dgetrf(int matrix_layout, int m, int n, double* a, int lda,
                   int* ipiv);
int LAPACKE_sgetri(int matrix_layout, int n, float* a, int lda,
                   const int* ipiv);
int LAPACKE_dgetri(int matrix_layout, int n, double* a, int lda,
                   const int* ipiv);
}
#endif

#include <cmath>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {

// Support continuous memory now
// If transA = N, and transB = N
// Then matrixA: M * K, matrixB: K * N, matrixC : M * N
// For more detailed info, please refer to
// http://www.netlib.org/lapack/explore-html/d4/de2/sgemm_8f.html

template <typename DeviceContext>
class Blas {
 public:
  explicit Blas(const DeviceContext& context) : context_(context) {}

  template <typename T>
  void GEMM(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
            const int M, const int N, const int K, const T alpha, const T* A,
            const T* B, const T beta, T* C) const;

  template <typename T>
  void GEMM(const bool transA, const bool transB, const int M, const int N,
            const int K, const T alpha, const T* A, const int lda, const T* B,
            const int ldb, const T beta, T* C, const int ldc) const;

 private:
  const DeviceContext& context_;
};

template <typename DeviceContext, typename T>
class BlasT : private Blas<DeviceContext> {
 public:
  using Blas<DeviceContext>::Blas;

  template <typename... ARGS>
  void GEMM(ARGS... args) const {
    static_cast<const Blas<DeviceContext>*>(this)->template GEMM<T>(args...);
  }
};

template <typename DeviceContext, typename T>
inline BlasT<DeviceContext, T> GetBlas(
    const framework::ExecutionContext& exe_ctx) {
  return BlasT<DeviceContext, T>(
      exe_ctx.template device_context<DeviceContext>());
}

template <typename DeviceContext, typename T>
inline BlasT<DeviceContext, T> GetBlas(const DeviceContext& dev_ctx) {
  return BlasT<DeviceContext, T>(dev_ctx);
}

// matrix multiply with continuous memory
template <typename DeviceContext, typename T>
void matmul(const DeviceContext& context, const framework::Tensor& matrix_a,
            bool trans_a, const framework::Tensor& matrix_b, bool trans_b,
            T alpha, framework::Tensor* matrix_out, T beta);

// Batched gemm
template <typename DeviceContext, typename T>
void batched_gemm(const DeviceContext& context, const CBLAS_TRANSPOSE transA,
                  const CBLAS_TRANSPOSE transB, const int M, const int N,
                  const int K, const T alpha, const T* A, const T* B,
                  const T beta, T* C, const int batchCount,
                  const int64_t strideA, const int64_t strideB);

template <typename DeviceContext, typename T>
void gemv(const DeviceContext& context, const bool trans_a, const int M,
          const int N, const T alpha, const T* A, const T* B, const T beta,
          T* C);

template <typename DeviceContext, typename T>
void axpy(const DeviceContext& context, const int n, const T alpha, const T* x,
          T* y);

template <typename DeviceContext, typename T, int Rank>
struct Transpose {
  void operator()(const DeviceContext& context, const framework::Tensor& in,
                  framework::Tensor* out, const std::vector<int>& axis);
};

template <typename DeviceContext, typename T>
struct SetConstant {
  void operator()(const DeviceContext& context, framework::Tensor* tensor,
                  T num);
};

template <typename Place>
void set_constant_with_place(const platform::DeviceContext& context,
                             framework::Tensor* tensor, float value);

void set_constant(const platform::DeviceContext& context,
                  framework::Tensor* tensor, float value);

template <typename DeviceContext, typename T>
struct RowwiseAdd {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& vec, framework::Tensor* output);
};

template <typename DeviceContext, typename T>
struct ColwiseSum {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseSum {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseMean {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle

#include "paddle/fluid/operators/math/blas_impl.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/operators/math/blas_impl.cu.h"
#endif
