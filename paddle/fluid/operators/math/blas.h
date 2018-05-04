//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"

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

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext>
class Blas {
 public:
  explicit Blas(const DeviceContext& context) : context_(context) {}

  template <typename T>
  void GEMM(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
            T alpha, const T* A, const T* B, T beta, T* C) const;

  template <typename T>
  void GEMM(bool transA, bool transB, int M, int N, int K, T alpha, const T* A,
            int lda, const T* B, int ldb, T beta, T* C, int ldc) const;

  template <typename T>
  void MatMul(const framework::Tensor& mat_a, bool trans_a,
              const framework::Tensor& mat_b, bool trans_b, T alpha,
              framework::Tensor* mat_out, T beta) const;

  template <typename T>
  void MatMul(const framework::Tensor& mat_a, bool trans_a,
              const framework::Tensor& mat_b, bool trans_b,
              framework::Tensor* mat_out) const {
    MatMul(mat_a, trans_a, mat_b, trans_b, static_cast<T>(1.0), mat_out,
           static_cast<T>(0.0));
  }

  template <typename T>
  void MatMul(const framework::Tensor& mat_a, const framework::Tensor& mat_b,
              framework::Tensor* mat_out) const {
    this->template MatMul<T>(mat_a, false, mat_b, false, mat_out);
  }

  template <typename T>
  void AXPY(int n, T alpha, const T* x, T* y) const;

  template <typename T>
  void GEMV(bool trans_a, int M, int N, T alpha, const T* A, const T* B, T beta,
            T* C) const;

  template <typename T>
  void BatchedGEMM(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N,
                   int K, T alpha, const T* A, const T* B, T beta, T* C,
                   int batchCount, int64_t strideA, int64_t strideB) const;

 private:
  const DeviceContext& context_;
};

template <typename DeviceContext, typename T>
class BlasT : private Blas<DeviceContext> {
 public:
  using Blas<DeviceContext>::Blas;

  template <typename... ARGS>
  void GEMM(ARGS... args) const {
    Base()->template GEMM<T>(args...);
  }

  template <typename... ARGS>
  void MatMul(ARGS... args) const {
    Base()->template MatMul<T>(args...);
  }

  template <typename... ARGS>
  void AXPY(ARGS... args) const {
    Base()->template AXPY<T>(args...);
  }

  template <typename... ARGS>
  void GEMV(ARGS... args) const {
    Base()->template GEMV<T>(args...);
  }

  template <typename... ARGS>
  void BatchedGEMM(ARGS... args) const {
    Base()->template BatchedGEMM<T>(args...);
  }

 private:
  const Blas<DeviceContext>* Base() const {
    return static_cast<const Blas<DeviceContext>*>(this);
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

}  // namespace math
}  // namespace operators
}  // namespace paddle

#include "paddle/fluid/operators/math/blas_impl.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/operators/math/blas_impl.cu.h"
#endif
