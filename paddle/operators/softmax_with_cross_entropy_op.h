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

#pragma once
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/cross_entropy_op.h"
#include "paddle/operators/math/softmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
class SoftmaxWithCrossEntropyKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(context.GetPlace()),
                   "This kernel only runs on CPU.");

    // Calculate ths softmax outputs.
    const Tensor* logits = context.Input<Tensor>("Logits");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    softmax->mutable_data<T>(context.GetPlace());

    math::SoftmaxFunctor<platform::CPUPlace, T>()(logits, softmax, context);

    // Calculate the cross entropy loss based on hard labels.
    T* softmax_out = softmax->data<T>();
    const int* label_data = context.Input<Tensor>("Label")->data<int>();

    Tensor* loss = context.Output<Tensor>("Loss");
    loss->mutable_data<T>(context.GetPlace());
    T* loss_data = loss->data<T>();

    const int batch_size = logits->dims()[0];
    const int class_num = logits->dims()[1];

    for (int i = 0; i < batch_size; ++i) {
      int index = i * class_num + label_data[i];
      loss_data[i] = -tolerable_value(std::log(softmax_out[index]));
    }
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    logit_grad->ShareDataWith<T>(*context.Input<Tensor>("Softmax"));
    T* logit_grad_data = logit_grad->data<T>();

    const int batch_size = logit_grad->dims()[0];
    const int class_num = logit_grad->dims()[1];

    const int* label_data = context.Input<Tensor>("Label")->data<int>();
    for (int i = 0; i < batch_size; ++i) {
      int index = i * class_num + label_data[i];
      logit_grad_data[index] -= 1.;
    }
  }
};

}  // namespace operators
}  // namespace paddle
