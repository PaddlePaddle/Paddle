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

#include "paddle/framework/op_registry.h"
#include "paddle/operators/cross_entropy_op.h"
#include "paddle/operators/math/softmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void CrossEntropyKernel(T* out, const T* softmax_out,
                                   const int* label, const int batch_size,
                                   const int class_num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    PADDLE_ASSERT(label[i] >= 0 && label[i] < class_num);
    out[i] = -tolerable_value(std::log(softmax_out[i * class_num + label[i]]));
  }
}

template <typename T>
__global__ void CrossEntropyWithSoftmaxGradKernel(T* softmax_out,
                                                  const int* label,
                                                  const int batch_size,
                                                  const int class_num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    PADDLE_ASSERT(label[i] >= 0 && label[i] < class_num);
    softmax_out[i * class_num + label[i]] -= 1.;
  }
}

template <typename T>
class SoftmaxWithCrossEntropyCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "This kernel only runs on GPU device.");

    // Calculate ths softmax outputs.
    const Tensor* logits = context.Input<Tensor>("Logits");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    softmax->mutable_data<T>(context.GetPlace());
    math::SoftmaxFunctor<platform::GPUPlace, T>()(logits, softmax, context);
    T* softmax_out = softmax->data<T>();

    // Calculate the cross entropy loss based on hard labels.
    const int* label_data = context.Input<Tensor>("Label")->data<int>();
    Tensor* loss = context.Output<Tensor>("Loss");
    loss->mutable_data<T>(context.GetPlace());
    T* loss_data = loss->data<T>();

    const int batch_size = logits->dims()[0];
    const int class_num = logits->dims()[1];
    int block = 512;
    int grid = (batch_size + block - 1) / block;

    CrossEntropyKernel<T><<<grid, block>>>(loss_data, softmax_out, label_data,
                                           batch_size, class_num);
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "This kernel only runs on GPU device.");

    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    logit_grad->ShareDataWith<T>(*context.Input<Tensor>("Softmax"));
    T* logit_grad_data = logit_grad->data<T>();

    const int batch_size = logit_grad->dims()[0];
    const int class_num = logit_grad->dims()[1];

    const int* label_data = context.Input<Tensor>("Label")->data<int>();

    const int block = 512;
    const int grid = (batch_size + block - 1) / block;

    CrossEntropyWithSoftmaxGradKernel<T><<<grid, block>>>(
        logit_grad_data, label_data, batch_size, class_num);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(softmax_with_cross_entropy,
                       ops::SoftmaxWithCrossEntropyCUDAKernel<float>);
REGISTER_OP_GPU_KERNEL(softmax_with_cross_entropy_grad,
                       ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>);
