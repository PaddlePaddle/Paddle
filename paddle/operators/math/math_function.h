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

#pragma once
#ifdef PADDLE_USE_MKLML
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <mkl_vml_functions.h>
#endif

#ifdef PADDLE_USE_MKL
#include <mkl.h>
#include <mkl_lapacke.h>
#endif

#ifdef PADDLE_USE_ATLAS
extern "C" {
#include <cblas.h>
#include <clapack.h>
}
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#include <lapacke.h>
#endif

#ifndef LAPACK_FOUND
extern "C" {
#include <cblas.h>
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

#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {

// Support continuous memory now
// If transA = N, and transB = N
// Then matrixA: M * K, matrixB: K * N, matrixC : M * N
// For more detailed info, please refer to
// http://www.netlib.org/lapack/explore-html/d4/de2/sgemm_8f.html
template <typename Place, typename T>
void gemm(const platform::DeviceContext& context, const CBLAS_TRANSPOSE transA,
          const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
          const T alpha, const T* A, const T* B, const T beta, T* C);

// gemm wrapper with stride args for matrix uncontinuous in memory
template <typename Place, typename T>
void gemm(const platform::DeviceContext& context, const bool transA,
          const bool transB, const int M, const int N, const int K,
          const T alpha, const T* A, const int lda, const T* B, const int ldb,
          const T beta, T* C, const int ldc);

// matrix multiply with continuous memory
template <typename Place, typename T>
void matmul(const platform::DeviceContext& context,
            const framework::Tensor& matrix_a, bool trans_a,
            const framework::Tensor& matrix_b, bool trans_b, T alpha,
            framework::Tensor* matrix_out, T beta);

// Batched gemm
template <typename Place, typename T>
void batched_gemm(const platform::DeviceContext& context,
                  const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
                  const int M, const int N, const int K, const T alpha,
                  const T* A, const T* B, const T beta, T* C,
                  const int batchCount, const int strideA, const int strideB);

// Implements the logic of numpy matmul:
// https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
//
// but allowing also for a, b to be transposed
//
// Both a & b can be 1- to 3-dimensional. Higher rank tensors are not supported
// yet.
template <typename Place, typename T>
class MatMulFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& a, bool trans_a,
                  const framework::Tensor& b, bool trans_b, T alpha,
                  framework::Tensor* out, T beta) {
    auto dim_a = a.dims();
    auto dim_b = b.dims();

    PADDLE_ENFORCE(a.place() == b.place() && b.place() == out->place(),
                   "Tensors must all be in the same place.");
    PADDLE_ENFORCE_GE(dim_a.size(), 1,
                      "Input tensor a must be at least 1-dimensional.");
    PADDLE_ENFORCE_GE(dim_b.size(), 1,
                      "Input tensor b must be at least 1-dimensional.");
    PADDLE_ENFORCE_LE(dim_a.size(), 3,
                      "Input tensor a must be at most 3-dimensional.");
    PADDLE_ENFORCE_LE(dim_b.size(), 3,
                      "Input tensor b must be at most 3-dimensional.");

    int M = 0, N = 0, kA = 0, kB = 0, batchCountA = 0, batchCountB = 0,
        strideA = 0, strideB = 0;

    switch (dim_a.size()) {
      case 1:
        // similar to np.matmul:
        // prepend dimension 1 (no transpose) or append dimension 1 (transpose)
        M = trans_a ? dim_a[0] : 1;
        kA = trans_a ? 1 : dim_a[0];
        break;
      case 2:
        M = trans_a ? dim_a[1] : dim_a[0];
        kA = trans_a ? dim_a[0] : dim_a[1];
        break;
      case 3:
        batchCountA = dim_a[0];
        M = trans_a ? dim_a[2] : dim_a[1];
        kA = trans_a ? dim_a[1] : dim_a[2];
        strideA = M * kA;
        break;
      default:
        assert(false);
    }

    switch (dim_b.size()) {
      case 1:
        // similar to np.matmul:
        // append dimension 1 (no transpose) or prepend dimension 1 (transpose)
        kB = trans_b ? 1 : dim_b[0];
        N = trans_b ? dim_b[0] : 1;
        break;
      case 2:
        kB = trans_b ? dim_b[1] : dim_b[0];
        N = trans_b ? dim_b[0] : dim_b[1];
        break;
      case 3:
        batchCountB = dim_b[0];
        kB = trans_b ? dim_b[2] : dim_b[1];
        N = trans_b ? dim_b[1] : dim_b[2];
        strideB = kB * N;
        break;
      default:
        assert(false);
    }

    PADDLE_ENFORCE_EQ(
        kA, kB,
        "First matrix's width must be equal with second matrix's height.");
    if (batchCountA && batchCountB) {
      PADDLE_ENFORCE_EQ(
          batchCountA, batchCountB,
          "When input tensors a and b are both batched, they must have the "
          "same batch dimension.");
    }
    int batchCount = std::max(batchCountA, batchCountB);

    CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

    if (!batchCount) {
      // regular matrix multiplication
      gemm<Place, T>(context, transA, transB, M, N, kA, alpha, a.data<T>(),
                     b.data<T>(), beta, out->data<T>());
    } else {
      // batched matrix multiplication
      batched_gemm<Place, T>(context, transA, transB, M, N, kA, alpha,
                             a.data<T>(), b.data<T>(), beta, out->data<T>(),
                             batchCount, strideA, strideB);
    }
  }
};

template <typename Place, typename T>
struct SetConstant {
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor* tensor, T num) {
    auto t = framework::EigenVector<T>::Flatten(*tensor);
    t.device(*context.GetEigenDevice<Place>()) =
        t.constant(static_cast<T>(num));
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
