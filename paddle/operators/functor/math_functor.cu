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

#include "paddle/operators/functor/math_functor.h"
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {
namespace functor {

template <typename T>
__global__ void SetKernel(const int N, const T alpha, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) { Y[i] = alpha; }
}

template <typename T>
struct Set<platform::GPUPlace, T> {
  void operator()(const T alpha, framework::Tensor* Y,
                  platform::DeviceContext* context) {
    int N = product(Y->dims());
    T* YData = Y->mutable_data<T>(context->GetPlace());
    SetKernel<<<(N + 512 - 1) / 512, 512>>>(N, alpha, YData);
  }
};

template struct Set<platform::GPUPlace, float>;
template struct Set<platform::GPUPlace, double>;

}  // namespace functor
}  // namespace operators
}  // namespace paddle
