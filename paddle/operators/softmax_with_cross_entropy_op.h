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
#include "paddle/operators/math/cross_entropy.h"
#include "paddle/operators/math/softmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
class SoftmaxWithCrossEntropyKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(context.GetPlace()),
                   "This kernel only runs on CPU.");
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    Tensor* loss = context.Output<Tensor>("Loss");

    softmax->mutable_data<T>(context.GetPlace());
    loss->mutable_data<T>(context.GetPlace());

    math::SoftmaxFunctor<platform::CPUPlace, T>()(context.device_context(),
                                                  logits, softmax);
    math::CrossEntropyFunctor<platform::CPUPlace, T>()(
        context.device_context(), loss, softmax, labels,
        context.Attr<bool>("softLabel"));
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Loss"));
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    logit_grad->ShareDataWith<T>(*context.Input<Tensor>("Softmax"));

    const int class_num = logit_grad->dims()[1];
    if (context.Attr<bool>("softLabel")) {
      auto out_grad_mat = EigenMatrix<T>::From(*out_grad);
      auto logit_grad_mat = EigenMatrix<T>::From(*logit_grad);
      auto lbl_mat = EigenMatrix<T>::From(*labels);

      logit_grad_mat.device(context.GetEigenDevice<platform::CPUPlace>()) =
          logit_grad_mat *
              out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, class_num)) -
          lbl_mat;
    } else {
      const int batch_size = logit_grad->dims()[0];
      const int* label_data = labels->data<int>();
      const T* out_grad_data = out_grad->data<T>();
      T* logit_grad_data = logit_grad->data<T>();

      for (int i = 0; i < batch_size; ++i) {
        int index = i * class_num + label_data[i];
        logit_grad_data[index] =
            (out_grad_data[i] * logit_grad_data[index] - 1.);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
