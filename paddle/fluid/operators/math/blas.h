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
#include <mkl_service.h>
#include <mkl_vml_functions.h>
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#ifdef LAPACK_FOUND
#include <lapacke.h>
#endif
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

static void SetNumThreads(int num_threads) {
#ifdef PADDLE_USE_OPENBLAS
  int real_num_threads = num_threads > 1 ? num_threads : 1;
  openblas_set_num_threads(real_num_threads);
#elif defined(PADDLE_WITH_MKLML)
  int real_num_threads = num_threads > 1 ? num_threads : 1;
  mkl_set_num_threads(real_num_threads);
#else
  PADDLE_ENFORCE(false, "To be implemented.");
#endif
}

/**
 * Matrix Descriptor of a memory buffer.
 *
 * It is used for Blas::MatMul. MatMul operator can be batched.
 * if Mat A is [BatchSize, H, W], Mat B is [BatchSize, H, W]. It will be a
 * `batch_size` times of GEMM. The batched GEMM could be faster base on the
 * implementation of the blas library. The batch size could be zero. If any
 * matrix of `matmul` has a batch size, the will be a batched GEMM, too. e.g.,
 * Mat A is [BatchSize, H1, W2], and Mat B [H2, W2], The result matrix wil be
 * [BatchSize, H1, W2]
 *
 * The boolean flag, `trans`, describe the memory is the transpose of matrix or
 * not. If the trans is true, the last two dims of matrix are transposed. The
 * memory layout of the matrix is [Width, Height] or [BatchSize, Width, Height].
 *
 * The MatDescriptor is not only the dimension or shape of a matrix, it also
 * contains the layout, stride of matrix. It is clearer to have a structure than
 * reuse `DDim`.
 */
struct MatDescriptor {
  int64_t height_;
  int64_t width_;
  int64_t stride_{0};
  int64_t batch_size_{0};
  bool trans_;
};

/**
 * Create Matrix Descriptor from a tensor dim, num_flatten_cols, and transpose
 * flag
 *
 * @param tensor_dim: The dimension of the tensor. The rank of this dimension
 * must larger than 1.
 *
 * @param num_flatten_cols:  Reshape a tensor to a matrix. The matrix's first
 * dimension(column length) will be the product of tensor's first `num_col_dims`
 * dimensions. If num_flatten_cols is zero, the first N-2 dimension will be the
 * batch_size of descriptor.
 *
 * @param trans: True if the matrix is transposed.
 */
extern MatDescriptor CreateMatrixDescriptor(const framework::DDim& tensor_dim,
                                            int num_flatten_cols, bool trans);

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
  void VADD(int n, const T* x, const T* y, T* z) const;

  template <typename T>
  void VCOPY(int n, const T* x, T* y) const;

  template <typename T>
  void GEMV(bool trans_a, int M, int N, T alpha, const T* A, const T* B, T beta,
            T* C) const;

  template <typename T>
  void BatchedGEMM(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N,
                   int K, T alpha, const T* A, const T* B, T beta, T* C,
                   int batchCount, int64_t strideA, int64_t strideB) const;

  template <typename T>
  void MatMul(const framework::Tensor& mat_a, const MatDescriptor& dim_a,
              const framework::Tensor& mat_b, const MatDescriptor& dim_b,
              T alpha, framework::Tensor* mat_out, T beta) const;

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
  void VADD(ARGS... args) const {
    Base()->template VADD<T>(args...);
  }

  template <typename... ARGS>
  void VCOPY(ARGS... args) const {
    Base()->template VCOPY<T>(args...);
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
