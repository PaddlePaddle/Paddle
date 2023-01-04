/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/fc_functor.h"

#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {
namespace funcs {

template <typename DeviceContext, typename T>
void FCFunctor<DeviceContext, T>::operator()(const DeviceContext& context,
                                             const int M,
                                             const int N,
                                             const int K,
                                             const T* X,
                                             const T* W,
                                             T* Y,
                                             const T* B,
                                             bool relu,
                                             bool padding_weights) {
  auto blas = GetBlas<DeviceContext, T>(context);
  phi::DenseTensor Y1;
  T* Y1_data = nullptr;
  if (padding_weights) {
    const int NN = N + 4;
    const int KK = K + 4;
    phi::DenseTensor X1;
    X1.Resize({M * KK});
    T* X1_data = context.template HostAlloc<T>(&X1);

    Y1.Resize({M * (N + 4)});
    Y1_data = context.template HostAlloc<T>(&Y1);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int i = 0; i < M; i++) {
      memcpy(X1_data + i * KK, X + i * K, K * sizeof(T));
    }
    blas.GEMM(false,
              false,
              M,
              N,
              K,
              static_cast<T>(1.0),
              X1_data,
              KK,
              W,
              NN,
              static_cast<T>(0.0),
              Y1_data,
              NN);
  } else {
    blas.MatMul(M, N, K, X, W, Y);
  }
  if (B == NULL) {
    if (padding_weights) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < M; i++) {
        memcpy(Y + i * N, Y1_data + i * (N + 4), N * sizeof(T));
      }
    }
    PADDLE_ENFORCE_EQ(
        relu,
        false,
        errors::PermissionDenied("When bias is NULL, relu can not be true."));
    return;
  }
  auto compute = relu ? paddle::operators::jit::KernelFuncs<
                            paddle::operators::jit::VAddReluTuple<T>,
                            paddle::platform::CPUPlace>::Cache()
                            .At(N)
                      : paddle::operators::jit::KernelFuncs<
                            paddle::operators::jit::VAddTuple<T>,
                            paddle::platform::CPUPlace>::Cache()
                            .At(N);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < M; i++) {
    T* dst = Y + i * N;
    T* src = (padding_weights) ? Y1_data + i * (N + 4) : dst;
    compute(B, src, dst, N);
  }
}

template class FCFunctor<CPUContext, float>;
template class FCFunctor<CPUContext, double>;

}  // namespace funcs
}  // namespace phi
