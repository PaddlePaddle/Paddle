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

#include "paddle/fluid/operators/math/blas.h"

DECLARE_int32(paddle_num_threads);

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext, typename T>
inline void FCCompute(const BlasT<DeviceContext, T>& blas, const int M,
                      const int N, const int K, const T* X, const T* W, T* Y,
                      const T* B = NULL) {
  blas.GEMM(CblasNoTrans, CblasNoTrans, M, N, K, static_cast<T>(1), X, W,
            static_cast<T>(0), Y);
  if (B) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for if (FLAGS_paddle_num_threads > 1)
#endif
    for (int i = 0; i < M; i++) {
      blas.AXPY(N, static_cast<T>(1), B, Y + i * N);
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
