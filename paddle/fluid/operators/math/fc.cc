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

#include "paddle/fluid/operators/math/fc.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class FCFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context, const int M,
                  const int N, const int K, const T* X, const T* W, T* Y,
                  const T* B = nullptr, bool relu = false,
                  bool weight_pass = false) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    framework::Tensor X1, Y1;
    X1.Resize({M * (K + 4)});
    T* X1_data = X1.mutable_data<T>(platform::CPUPlace());
    Y1.Resize({M * (N + 4)});
    T* Y1_data = Y1.mutable_data<T>(platform::CPUPlace());
    if (weight_pass) {
      const int NN = N + 4;
      const int KK = K + 4;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < M; i++) {
        memcpy(X1_data + i * KK, X + i * K, K * sizeof(X[0]));
      }
      blas.GEMM(false, false, M, N, K, static_cast<T>(1.0), X1_data, KK, W, NN,
                static_cast<T>(0.0), Y1_data, NN);
    } else {
      blas.MatMul(M, N, K, X, W, Y);
    }
    if (B == NULL) {
      return;
    }
    if (relu) {
      auto compute =
          jit::KernelFuncs<jit::VAddReluTuple<T>, platform::CPUPlace>::Cache()
              .At(N);
      for (int i = 0; i < M; i++) {
        T* dst = Y + i * N;
        T* src = nullptr;
        if (weight_pass)
          src = Y1_data + i * (N + 4);
        else
          src = dst;
        compute(B, src, dst, N);
      }
    } else {
      auto compute =
          jit::KernelFuncs<jit::VAddTuple<T>, platform::CPUPlace>::Cache().At(
              N);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < M; i++) {
        T* dst = Y + i * N;
        T* src = nullptr;
        if (weight_pass)
          src = Y1_data + i * (N + 4);
        else
          src = dst;
        compute(B, src, dst, N);
      }
    }
  }
};

template class FCFunctor<platform::CPUDeviceContext, float>;
template class FCFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
