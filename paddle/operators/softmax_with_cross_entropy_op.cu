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

#define EIGEN_USE_GPU

#include "paddle/operators/softmax_with_cross_entropy_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

namespace {
template <typename T>
__global__ void CrossEntropyGrad(T* out_grad, const T* in_grad,
                                 const int* labels, const int batch_size,
                                 const int class_num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_idx = tid / class_num;

  if (tid < batch_size * class_num) out_grad[tid] *= in_grad[sample_idx];
  __syncthreads();

  if (tid < batch_size) {
    PADDLE_ASSERT(labels[sample_idx] >= 0 && labels[sample_idx] < class_num);
    out_grad[tid * class_num + labels[tid]] -= 1.;
  }
}

template <typename T>
__global__ void SoftCrossEntropyGradientKernel(T* logit_grad,
                                               const T* loss_grad,
                                               const T* labels,
                                               const int batch_size,
                                               const int class_num) {
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < batch_size * class_num) {
    int row_ids = ids / class_num;
    logit_grad[ids] = logit_grad[ids] * loss_grad[row_ids] - labels[ids];
  }
}
}  // namespace

template <typename T>
class SoftmaxWithCrossEntropyCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "This kernel only runs on GPU device.");
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");

    Tensor* loss = context.Output<Tensor>("Loss");
    softmax->mutable_data<T>(context.GetPlace());
    loss->mutable_data<T>(context.GetPlace());

    math::SoftmaxFunctor<platform::GPUPlace, T>()(context.device_context(),
                                                  logits, softmax);
    math::CrossEntropyFunctor<platform::GPUPlace, T>()(
        context.device_context(), loss, softmax, labels,
        context.Attr<bool>("softLabel"));
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "This kernel only runs on GPU device.");
    const Tensor* labels = context.Input<Tensor>("Label");
    const T* loss_grad_data =
        context.Input<Tensor>(framework::GradVarName("Loss"))->data<T>();
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    logit_grad->ShareDataWith<T>(*context.Input<Tensor>("Softmax"));
    T* logit_grad_data = logit_grad->data<T>();

    const int batch_size = logit_grad->dims()[0];
    const int class_num = logit_grad->dims()[1];
    int block = 512;
    int grid = (batch_size * class_num + block - 1) / block;

    if (context.Attr<bool>("softLabel")) {
      const T* label_data = labels->data<T>();
      SoftCrossEntropyGradientKernel<T><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              context.device_context())
                              .stream()>>>(logit_grad_data, loss_grad_data,
                                           label_data, batch_size, class_num);
    } else {
      const int* label_data = labels->data<int>();
      CrossEntropyGrad<T><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              context.device_context())
                              .stream()>>>(logit_grad_data, loss_grad_data,
                                           label_data, batch_size, class_num);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(softmax_with_cross_entropy,
                       ops::SoftmaxWithCrossEntropyCUDAKernel<float>);
REGISTER_OP_GPU_KERNEL(softmax_with_cross_entropy_grad,
                       ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>);
