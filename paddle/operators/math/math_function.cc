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

#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <>
void gemm<platform::CPUPlace, float>(const CBLAS_TRANSPOSE transA,
                                     const CBLAS_TRANSPOSE transB,
                                     const int M,
                                     const int N,
                                     const int K,
                                     const float alpha,
                                     const float* A,
                                     const int lda,
                                     const float* B,
                                     const int ldb,
                                     const float beta,
                                     float* C,
                                     const int ldc,
                                     platform::DeviceContext* context) {
  cblas_sgemm(CblasRowMajor,
              transA,
              transB,
              M,
              N,
              K,
              alpha,
              A,
              lda,
              B,
              ldb,
              beta,
              C,
              ldc);
}

template <>
void gemm<platform::CPUPlace, double>(const CBLAS_TRANSPOSE transA,
                                      const CBLAS_TRANSPOSE transB,
                                      const int M,
                                      const int N,
                                      const int K,
                                      const double alpha,
                                      const double* A,
                                      const int lda,
                                      const double* B,
                                      const int ldb,
                                      const double beta,
                                      double* C,
                                      const int ldc,
                                      platform::DeviceContext* context) {
  cblas_dgemm(CblasRowMajor,
              transA,
              transB,
              M,
              N,
              K,
              alpha,
              A,
              lda,
              B,
              ldb,
              beta,
              C,
              ldc);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
