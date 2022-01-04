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

#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__global__ void CrossEntropyKernel(T* Y, const T* X, const int64_t* label,
                                   const int N, const int D,
                                   const int ignore_index) {
  CUDA_KERNEL_LOOP(i, N) {
    PADDLE_ENFORCE(label[i] >= 0 && label[i] < D || label[i] == ignore_index,
                   "The value of label[%d] expected >= 0 and < %ld, or == %ld, "
                   "but got %ld. Please check input value.",
                   i, D, ignore_index, label[i]);
    Y[i] = ignore_index == label[i]
               ? static_cast<T>(0)
               : -math::TolerableValue<T>()(real_log(X[i * D + label[i]]));
  }
}

template <typename T>
__global__ void SoftCrossEntropyKernel(T* Y, const T* X, const T* label,
                                       const int class_num) {
  int tid = threadIdx.x;
  T val(0);

  int idx = blockIdx.x * class_num + tid;
  int end = blockIdx.x * class_num + class_num;
  for (; idx < end; idx += blockDim.x) {
    val += math::TolerableValue<T>()(real_log(X[idx])) * label[idx];
  }

  val = paddle::platform::reduceSum(val, tid, blockDim.x);
  if (threadIdx.x == 0) {
    Y[blockIdx.x] = -val;
  }
}

template <typename T>
class CrossEntropyFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& ctx,
                  framework::Tensor* out, const framework::Tensor* prob,
                  const framework::Tensor* labels, const bool softLabel,
                  const int ignore_index, const int axis_dim) {
    const T* prob_data = prob->data<T>();
    T* loss_data = out->mutable_data<T>(ctx.GetPlace());

    int batch_size = prob->dims()[0];
    int class_num = prob->dims()[1];
#ifdef __HIPCC__
    constexpr int kMaxBlockDim = 256;
#else
    constexpr int kMaxBlockDim = 512;
#endif

    if (softLabel) {
      const T* label_data = labels->data<T>();
      int block = class_num > kMaxBlockDim
                      ? kMaxBlockDim
                      : pow(2, static_cast<int>(std::log2(class_num)));

      SoftCrossEntropyKernel<T><<<batch_size, block, 0, ctx.stream()>>>(
          loss_data, prob_data, label_data, class_num);
    } else {
      const int64_t* label_data = labels->data<int64_t>();
      int block = kMaxBlockDim;
      int grid = (batch_size + block - 1) / block;
      CrossEntropyKernel<T><<<grid, block, 0, ctx.stream()>>>(
          loss_data, prob_data, label_data, batch_size, class_num,
          ignore_index);
    }
  }
};

template class CrossEntropyFunctor<platform::CUDADeviceContext, float>;
template class CrossEntropyFunctor<platform::CUDADeviceContext, double>;
template class CrossEntropyFunctor<platform::CUDADeviceContext,
                                   platform::float16>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
