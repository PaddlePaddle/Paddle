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

#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, typename LabelT>
__global__ void CrossEntropyKernel(T* Y,
                                   const T* X,
                                   const LabelT* label,
                                   const int N,
                                   const int D,
                                   const int ignore_index) {
  CUDA_KERNEL_LOOP(i, N) {
    auto lbl = static_cast<int64_t>(label[i]);
    PADDLE_ENFORCE(lbl >= 0 && lbl < D || lbl == ignore_index,
                   "The value of label[%d] expected >= 0 and < %ld, or == %ld, "
                   "but got %ld. Please check input value.",
                   i,
                   D,
                   ignore_index,
                   lbl);
    Y[i] = ignore_index == lbl
               ? static_cast<T>(0)
               : -math::TolerableValue<T>()(real_log(X[i * D + lbl]));
  }
}

template <typename T>
__global__ void SoftCrossEntropyKernel(T* Y,
                                       const T* X,
                                       const T* label,
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
struct HardLabelCrossEntropyCUDAFunctorImpl {
 public:
  HardLabelCrossEntropyCUDAFunctorImpl(T* loss_data,
                                       const T* prob_data,
                                       const void* label_data,
                                       const int batch_size,
                                       const int class_num,
                                       const int ignore_index,
                                       const int block_size,
                                       gpuStream_t stream)
      : loss_data_(loss_data),
        prob_data_(prob_data),
        label_data_(label_data),
        batch_size_(batch_size),
        class_num_(class_num),
        ignore_index_(ignore_index),
        block_size_(block_size),
        stream_(stream) {}

  template <typename U>
  void apply() const {
    int grid_size = (batch_size_ + block_size_ - 1) / block_size_;
    CrossEntropyKernel<T, U><<<grid_size, block_size_, 0, stream_>>>(
        loss_data_,
        prob_data_,
        static_cast<const U*>(label_data_),
        batch_size_,
        class_num_,
        ignore_index_);
  }

 private:
  T* loss_data_;
  const T* prob_data_;
  const void* label_data_;
  const int batch_size_;
  const int class_num_;
  const int ignore_index_;
  const int block_size_;
  gpuStream_t stream_;
};

template <typename DeviceContext, typename T>
void CrossEntropyFunctor<DeviceContext, T>::operator()(
    const DeviceContext& ctx,
    phi::DenseTensor* out,
    const phi::DenseTensor* prob,
    const phi::DenseTensor* labels,
    const bool softLabel,
    const int ignore_index,
    const int axis_dim) {
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
    HardLabelCrossEntropyCUDAFunctorImpl<T> functor(loss_data,
                                                    prob_data,
                                                    labels->data(),
                                                    batch_size,
                                                    class_num,
                                                    ignore_index,
                                                    kMaxBlockDim,
                                                    ctx.stream());
    framework::VisitDataType(framework::TransToProtoVarType(labels->dtype()),
                             functor);
  }
}

template class CrossEntropyFunctor<phi::GPUContext, float>;
template class CrossEntropyFunctor<phi::GPUContext, double>;
template class CrossEntropyFunctor<phi::GPUContext, platform::float16>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
