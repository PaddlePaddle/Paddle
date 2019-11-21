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

#include <algorithm>
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/fc.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, bool DoRelu>
__global__ void InplaceAddReluKernel(const T* bias, T* data, int M, int N) {
  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    int index = i * N + threadIdx.x;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      T tmp = data[index] + bias[j];
      if (DoRelu) {
        data[index] = (tmp > 0) ? tmp : 0;
      } else {
        data[index] = tmp;
      }
      index += blockDim.x;
    }
  }
}

template <typename T>
class FCFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context, const int M,
                  const int N, const int K, const T* X, const T* W, T* Y,
                  const T* B = nullptr, bool relu = false,
                  bool padding_weights = false) {
    PADDLE_ENFORCE_EQ(
        padding_weights, false,
        platform::errors::PermissionDenied(
            "Weight padding in fc can not be used in GPU scope."));
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(context);
    blas.GEMM(false, false, M, N, K, static_cast<T>(1.0), X, K, W, N,
              static_cast<T>(0.0), Y, N);
    if (B == NULL) {
      return;
    }

    const int kThreadsPerBlock = 1024;
    int max_threads = context.GetMaxPhysicalThreadCount();
    int num_threads = std::min(kThreadsPerBlock, (((N + 31) >> 5) << 5));
    int num_blocks = std::max(max_threads / num_threads, 1);
    if (relu) {
      InplaceAddReluKernel<
          T, true><<<num_blocks, num_threads, 0, context.stream()>>>(B, Y, M,
                                                                     N);
    } else {
      InplaceAddReluKernel<
          T, false><<<num_blocks, num_threads, 0, context.stream()>>>(B, Y, M,
                                                                      N);
    }
  }
};

template class FCFunctor<platform::CUDADeviceContext, float>;
template class FCFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
