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

#define EIGEN_USE_GPU

#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/math/softmax_impl.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__global__ void softmax_kernel(const int batch_size, const int num_classes,
                               const T* logits, const int n, T* softmax) {
  extern __shared__ T shms[];
  extern __shared__ T exp_vals[];
  extern __shared__ T logits_buffer;

  const T kThreshold = -64;

  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
    int batch_id = idx / batch_size;
    shms[batch_id] = kThreshold;
    for (int class_id = 0; class_id < num_classes; ++class_id) {
      if (logits[idx] > shms[idx] && logits[idx]) {
      }
    }
    exp_vals[idx] = __expf(logits[idx] - shms[idx]);
  }
  __syncthreads();

  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
  }
}

template <typename T>
__global__ void softmax_gradient_kernel(const int batch_size,
                                        const int num_classes, const T* Y,
                                        const T* dY, T* dX) {
  extern __shared__ float reduction_buffer[];

  Y += blockIdx.x * dim;
  dY += blockIdx.x * dim;
  dX += blockIdx.x * dim;
  const int idx = threadIdx.x;
  float tmp;
  // A two-level reduction to compute the inner products.
  tmp = 0;
  for (int i = idx; i < dim; i += blockDim.x) {
    tmp += dY[i] * Y[i];
  }
  reduction_buffer[idx] = tmp;
  __syncthreads();
  if (idx == 0) {
    tmp = reduction_buffer[0];
    for (int i = 1; i < blockDim.x; ++i) tmp += reduction_buffer[i];
    reduction_buffer[0] = tmp;
  }
  __syncthreads();
  // Compute gradient.
  tmp = reduction_buffer[0];
  for (int i = idx; i < dim; i += blockDim.x) {
    dX[i] = Y[i] * (dY[i] - tmp);
  }
}

template <typename T>
void SoftmaxFunctor<platform::CUDADeviceContext, T>::operator()(
    const DeviceContext& context, const framework::Tensor* X,
    framework::Tensor* Y) {
  const T* logits = X->data<T>();
  T* softmax = Y->data<T>();

  const int batch_size = X->dims()[0];
  const int num_classes = X->dims()[1];
  const int n = static_cast<int>(X->numel());

  softmax_kernel<>
}

template class SoftmaxFunctor<platform::CUDADeviceContext, float>;
template class SoftmaxFunctor<platform::CUDADeviceContext, double>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext, float>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
