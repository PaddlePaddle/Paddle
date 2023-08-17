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

#if defined(__MUSACC__)
#include <thrust/device_vector.h>
#endif
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/kernels/funcs/math_function.h"

PHI_DECLARE_bool(enable_cublas_tensor_op_math);
PHI_DECLARE_bool(gemm_use_half_precision_compute_type);

namespace phi {
namespace funcs {

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int M,
                                 int N,
                                 int K,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::float16 alpha,
                                        const phi::dtype::float16 *A,
                                        const phi::dtype::float16 *B,
                                        phi::dtype::float16 beta,
                                        phi::dtype::float16 *C) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::bfloat16 alpha,
                                        const phi::dtype::bfloat16 *A,
                                        const phi::dtype::bfloat16 *B,
                                        phi::dtype::bfloat16 beta,
                                        phi::dtype::bfloat16 *C) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::complex<float> alpha,
                                        const phi::dtype::complex<float> *A,
                                        const phi::dtype::complex<float> *B,
                                        phi::dtype::complex<float> beta,
                                        phi::dtype::complex<float> *C) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::complex<double> alpha,
                                        const phi::dtype::complex<double> *A,
                                        const phi::dtype::complex<double> *B,
                                        phi::dtype::complex<double> beta,
                                        phi::dtype::complex<double> *C) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMM(bool transA,
                                 bool transB,
                                 int M,
                                 int N,
                                 int K,
                                 T alpha,
                                 const T *A,
                                 int lda,
                                 const T *B,
                                 int ldb,
                                 T beta,
                                 T *C,
                                 int ldc) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(bool transA,
                                        bool transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::float16 alpha,
                                        const phi::dtype::float16 *A,
                                        int lda,
                                        const phi::dtype::float16 *B,
                                        int ldb,
                                        phi::dtype::float16 beta,
                                        phi::dtype::float16 *C,
                                        int ldc) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(bool transA,
                                        bool transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::bfloat16 alpha,
                                        const phi::dtype::bfloat16 *A,
                                        int lda,
                                        const phi::dtype::bfloat16 *B,
                                        int ldb,
                                        phi::dtype::bfloat16 beta,
                                        phi::dtype::bfloat16 *C,
                                        int ldc) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::AXPY(int n, T alpha, const T *x, T *y) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::SCAL(int n, const T alpha, T *x) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::VCOPY(int n, const T *x, T *y) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                 int M,
                                 int N,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                        int M,
                                        int N,
                                        phi::dtype::float16 alpha,
                                        const phi::dtype::float16 *A,
                                        const phi::dtype::float16 *B,
                                        phi::dtype::float16 beta,
                                        phi::dtype::float16 *C) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                        int M,
                                        int N,
                                        phi::dtype::bfloat16 alpha,
                                        const phi::dtype::bfloat16 *A,
                                        const phi::dtype::bfloat16 *B,
                                        phi::dtype::bfloat16 beta,
                                        phi::dtype::bfloat16 *C) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        T alpha,
                                        const T *A,
                                        const T *B,
                                        T beta,
                                        T *C,
                                        int batchCount,
                                        int64_t strideA,
                                        int64_t strideB) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::dtype::bfloat16 alpha,
                                               const phi::dtype::bfloat16 *A,
                                               const phi::dtype::bfloat16 *B,
                                               phi::dtype::bfloat16 beta,
                                               phi::dtype::bfloat16 *C,
                                               int batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        T alpha,
                                        const T **A,
                                        const T **B,
                                        T beta,
                                        T **C,
                                        int batchCount) const {}

#if defined(__MUSACC__)
template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               double alpha,
                                               const double **A,
                                               const double **B,
                                               double beta,
                                               double **C,
                                               int batchCount) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               float alpha,
                                               const float **A,
                                               const float **B,
                                               float beta,
                                               float **C,
                                               int batchCount) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::dtype::float16 alpha,
                                               const phi::dtype::float16 **A,
                                               const phi::dtype::float16 **B,
                                               phi::dtype::float16 beta,
                                               phi::dtype::float16 **C,
                                               int batchCount) const {}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::dtype::bfloat16 alpha,
                                               const phi::dtype::bfloat16 **A,
                                               const phi::dtype::bfloat16 **B,
                                               phi::dtype::bfloat16 beta,
                                               phi::dtype::bfloat16 **C,
                                               int batchCount) const {}
#endif

template <>
template <typename T>
void Blas<phi::GPUContext>::TRSM(CBLAS_SIDE side,
                                 CBLAS_UPLO uplo,
                                 CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag,
                                 int M,
                                 int N,
                                 T alpha,
                                 const T *A,
                                 int lda,
                                 T *B,
                                 int ldb) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRF(
    int n, T **a, int *ipiv, int *info, int batch_size) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRI(int n,
                                         const T **a,
                                         const int *ipiv,
                                         T **a_inv,
                                         int *info,
                                         int batch_size) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedMatInv(
    int n, const T **a, T **a_inv, int *info, int batch_size) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRS(CBLAS_TRANSPOSE trans,
                                         int n,
                                         int nrhs,
                                         const T **a,
                                         int lda,
                                         int *ipiv,
                                         T **b,
                                         int ldb,
                                         int *info,
                                         int batch_size) const {}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedTRSM(CBLAS_SIDE side,
                                        CBLAS_UPLO uplo,
                                        CBLAS_TRANSPOSE transA,
                                        CBLAS_DIAG diag,
                                        int M,
                                        int N,
                                        T alpha,
                                        const T **A,
                                        int lda,
                                        T **B,
                                        int ldb,
                                        int batch_size) const {}

}  // namespace funcs
}  // namespace phi
