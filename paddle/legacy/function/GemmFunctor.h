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

#include "TensorType.h"

namespace paddle {

// TODO(hedaoyuan): Since the hl_matrix_mul interface does not conform to the
// cblas_dgemm interface's parameter format, it is necessary to introduce
// GemmFunctor as a new interface. Later, when considering the implementation
// of MatMulFunction, we need to consider the reconstruction of hl_matrix_mul
// interface.
template <DeviceType Device, class T>
struct BlasGemm {
  static void compute(const bool transA,
                      const bool transB,
                      const int M,
                      const int N,
                      const int K,
                      const T alpha,
                      const T* A,
                      const int lda,
                      const T* B,
                      const int ldb,
                      const T beta,
                      T* C,
                      const int ldc);
};

// TODO(hedaoyuan): Since the definition of the real type in the Paddle
// conflicts with the Eigen library, so compile the Eigen code can not
// include the Paddle header file. And need an EigenBlasGemm template class
// that does not contain the DeviceType parameter.
// I will fix this problem and merge BlasGemm and EigenBlasGemm into one.
template <class T>
struct EigenBlasGemm {
  static void compute(const bool transA,
                      const bool transB,
                      const int M,
                      const int N,
                      const int K,
                      const T alpha,
                      const T* A,
                      const int lda,
                      const T* B,
                      const int ldb,
                      const T beta,
                      T* C,
                      const int ldc);
};

}  // namespace paddle
