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

#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {

// Support continuous memory now
// If transA = N, and transB = N
// Then matrixA: M * K, matrixB: K * N matrixC : M * N
// For more detailed info, please refer to
// http://www.netlib.org/lapack/explore-html/d4/de2/sgemm_8f.html
template <typename Place, typename T>
void gemm(const platform::DeviceContext& context, const CBLAS_TRANSPOSE transA,
          const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
          const T alpha, const T* A, const T* B, const T beta, T* C);

// matrix multiply with continuous memory
template <typename Place, typename T>
void matmul(const platform::DeviceContext& context,
            const framework::Tensor& matrix_a, bool trans_a,
            const framework::Tensor& matrix_b, bool trans_b, T alpha,
            framework::Tensor* matrix_out, T beta);

}  // namespace math
}  // namespace operators
}  // namespace paddle
