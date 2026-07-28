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

#include <limits>

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/jit/kernels.h"

namespace phi {
namespace funcs {

template <typename DeviceContext, typename T>
void FCFunctor<DeviceContext, T>::operator()(const DeviceContext& dev_ctx,
                                             int M,
                                             int N,
                                             int K,
                                             const T* X,
                                             const T* W,
                                             T* Y,
                                             const T* B,
                                             bool relu,
                                             bool padding_weights) {
  (*this)(dev_ctx,
          static_cast<int64_t>(M),
          static_cast<int64_t>(N),
          static_cast<int64_t>(K),
          X,
          W,
          Y,
          B,
          relu,
          padding_weights);
}

template <typename DeviceContext, typename T>
void FCFunctor<DeviceContext, T>::operator()(const DeviceContext& dev_ctx,
                                             int64_t M,
                                             int64_t N,
                                             int64_t K,
                                             const T* X,
                                             const T* W,
                                             T* Y,
                                             const T* B,
                                             bool relu,
                                             bool padding_weights) {
  detail::to_blas_int(M, "FC CPU row count");
  detail::to_blas_int(N, "FC CPU output width");
  detail::to_blas_int(K, "FC CPU input width");
  if (padding_weights) {
    constexpr int64_t kMaxPaddedDimension =
        std::numeric_limits<int>::max() - 4LL;
    PADDLE_ENFORCE_LE(
        N,
        kMaxPaddedDimension,
        errors::InvalidArgument(
            "FC CPU padded output width exceeds INT_MAX, but received %ld.",
            N));
    PADDLE_ENFORCE_LE(
        K,
        kMaxPaddedDimension,
        errors::InvalidArgument(
            "FC CPU padded input width exceeds INT_MAX, but received %ld.", K));
  }
  int bias_width = 0;
  if (B != nullptr) {
    bias_width = static_cast<int>(N);
  }
  auto blas = GetBlas<DeviceContext, T>(dev_ctx);
  DenseTensor Y1;
  T* Y1_data = nullptr;
  if (padding_weights) {
    const int64_t NN = N + 4;
    const int64_t KK = K + 4;
    DenseTensor X1;
    X1.Resize({M * KK});
    T* X1_data = dev_ctx.template HostAlloc<T>(&X1);

    Y1.Resize({M * (N + 4)});
    Y1_data = dev_ctx.template HostAlloc<T>(&Y1);
#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < M; i++) {
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
  if (B == nullptr) {
    if (padding_weights) {
#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
#pragma omp parallel for
#endif
      for (int64_t i = 0; i < M; i++) {
        memcpy(Y + i * N, Y1_data + i * (N + 4), N * sizeof(T));
      }
    }
    PADDLE_ENFORCE_EQ(
        relu,
        false,
        errors::PermissionDenied("When bias is NULL, relu can not be true."));
    return;
  }
  if (M == 0 || N == 0) {
    return;
  }
  auto compute =
      relu
          ? phi::jit::KernelFuncs<phi::jit::VAddReluTuple<T>, CPUPlace>::Cache()
                .At(bias_width)
          : phi::jit::KernelFuncs<phi::jit::VAddTuple<T>, CPUPlace>::Cache().At(
                bias_width);
#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < M; i++) {
    T* dst = Y + i * N;
    T* src = (padding_weights) ? Y1_data + i * (N + 4) : dst;
    compute(B, src, dst, bias_width);
  }
}

template class FCFunctor<CPUContext, float>;
template class FCFunctor<CPUContext, double>;

}  // namespace funcs
}  // namespace phi
