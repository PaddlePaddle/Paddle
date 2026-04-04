// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once
/*
  Provides a subset of CUDA BLAS functions as templates:

    gemm<Dtype>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
  ldc)

  where Dtype is double, float, c10::complex<double>, c10::complex<float>,
  at::Half or at::BFloat16. The functions are available in at::cuda::blas
  namespace.
 */

#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>

namespace at::cuda::blas {

/* LEVEL 3 BLAS FUNCTIONS */

#define CUDABLAS_GEMM_ARGTYPES(Dtype) \
  CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, Dtype)

#define CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)                  \
  char transa, char transb, int64_t m, int64_t n, int64_t k,                \
      at::opmath_type<Dtype> alpha, const Dtype *a, int64_t lda,            \
      const Dtype *b, int64_t ldb, at::opmath_type<Dtype> beta, C_Dtype *c, \
      int64_t ldc

#define CUDABLAS_GEMM_ARGS(Dtype) \
  transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc

template <typename Dtype, typename C_Dtype = Dtype>
inline void gemm(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  static_assert(false && sizeof(Dtype),
                "at::cuda::blas::gemm: not implemented");
}

template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double));
template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float));
template <>
void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>));
template <>
void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>));
template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half));
template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16));

}  // namespace at::cuda::blas
