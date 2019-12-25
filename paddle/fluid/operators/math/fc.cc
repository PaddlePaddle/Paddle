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
#include "paddle/fluid/platform/parallel.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class FCFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context, const int M,
                  const int N, const int K, const T* X, const T* W, T* Y,
                  const T* B = nullptr, bool relu = false,
                  bool padding_weights = false) {
    if (!B) {
      PADDLE_ENFORCE_EQ(relu, false,
                        platform::errors::PermissionDenied(
                            "When bias is NULL, relu can not be true."));
    }

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    T* Y_padding = nullptr;

    auto compute =
        relu
            ? jit::KernelFuncs<jit::VAddReluTuple<T>,
                               platform::CPUPlace>::Cache()
                  .At(N)
            : jit::KernelFuncs<jit::VAddTuple<T>, platform::CPUPlace>::Cache()
                  .At(N);
    auto parallel_compute = [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; i++) {
        T* dst = Y + i * N;
        T* src = Y_padding ? Y_padding + i * (N + 4) : dst;
        compute(B, src, dst, N);
      }
    };

    // Because of the overhead of memcpy, we only do padding for GEMM
    //  when weights is already padded in fc_fuse_pass.
    if (padding_weights) {
      const int NN = N + 4;
      const int KK = K + 4;

      // NOTE: here need to mutable_data for temporary Tensor X_padding_tensor
      //  and Y_padding_tensor, the overhead is unmeasured.
      framework::Tensor X_padding_tensor;
      T* X_padding =
          X_padding_tensor.mutable_data<T>({M * KK}, platform::CPUPlace());

      framework::Tensor Y_padding_tensor;
      Y_padding =
          Y_padding_tensor.mutable_data<T>({M * (N + 4)}, platform::CPUPlace());

      auto parallel_memcpy_x = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          memcpy(X_padding + i * KK, X + i * K, K * sizeof(T));
        }
      };
      platform::RunParallelFor(0, M, parallel_memcpy_x);

      blas.GEMM(false, false, M, N, K, static_cast<T>(1.0), X_padding, KK, W,
                NN, static_cast<T>(0.0), Y_padding, NN);

      if (!B) {
        auto parallel_memcpy_y = [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; i++) {
            memcpy(Y + i * N, Y_padding + i * (N + 4), N * sizeof(T));
          }
        };
        platform::RunParallelFor(0, M, parallel_memcpy_y);
        return;
      }

      platform::RunParallelFor(0, M, parallel_compute);
    } else {
      blas.MatMul(M, N, K, X, W, Y);
      if (!B) {
        return;
      }

      platform::RunParallelFor(0, M, parallel_compute);
    }
  }
};

template class FCFunctor<platform::CPUDeviceContext, float>;
template class FCFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
