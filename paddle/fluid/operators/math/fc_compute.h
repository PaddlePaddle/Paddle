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

#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext, typename T>
inline void FCCompute(const BlasT<DeviceContext, T>& blas, const int M,
                      const int N, const int K, const T* X, const T* W, T* Y,
                      const T* B = NULL, bool relu = false) {
  blas.MatMul(M, N, K, X, W, Y);
  if (B == NULL) {
    return;
  }
  if (relu) {
    auto compute =
        jit::KernelFuncs<jit::VAddReluTuple<T>, platform::CPUPlace>::Cache().At(
            N);
    for (int i = 0; i < M; i++) {
      T* dst = Y + i * N;
      compute(B, dst, dst, N);
    }
  } else {
    auto compute =
        jit::KernelFuncs<jit::VAddTuple<T>, platform::CPUPlace>::Cache().At(N);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int i = 0; i < M; i++) {
      T* dst = Y + i * N;
      compute(B, dst, dst, N);
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
