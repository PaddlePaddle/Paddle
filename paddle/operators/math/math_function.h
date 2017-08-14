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
void gemm(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
          const int M, const int N, const int K, const T alpha, const T* A,
          const T* B, const T beta, T* C, platform::DeviceContext* context);

// matrix multiply with continuous memory
template <typename Place, typename T>
void matmul(const framework::Tensor& matrix_a, bool trans_a,
            const framework::Tensor& matrix_b, bool trans_b, float alpha,
            framework::Tensor* matrix_out, float beta,
            platform::DeviceContext* context);

}  // namespace math
}  // namespace operators
}  // namespace paddle
