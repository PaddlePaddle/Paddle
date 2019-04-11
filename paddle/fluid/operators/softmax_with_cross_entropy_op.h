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

#pragma once
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/softmax.h"

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

    int axis_dim = logits->dims()[logits->dims().size() - 1];

    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    math::SoftmaxFunctor<platform::CPUDeviceContext, T, false>()(
        dev_ctx, axis_dim, logits, softmax);
    math::CrossEntropyFunctor<platform::CPUDeviceContext, T>()(
        dev_ctx, loss, softmax, labels, context.Attr<bool>("soft_label"),
        context.Attr<int>("ignore_index"));
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
    logit_grad->ShareDataWith(*context.Input<Tensor>("Softmax"));

    const int class_num = logit_grad->dims()[1];
    auto out_grad_mat = EigenMatrix<T>::From(*out_grad);
    auto logit_grad_mat = EigenMatrix<T>::From(*logit_grad);
    auto& place = *context.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    if (context.Attr<bool>("soft_label")) {
      auto lbl_mat = EigenMatrix<T>::From(*labels);
      logit_grad_mat.device(place) =
          out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, class_num)) *
          (logit_grad_mat - lbl_mat);
    } else {
      logit_grad_mat.device(place) =
          logit_grad_mat *
          out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, class_num));

      const int batch_size = logit_grad->dims()[0];
      const int64_t* label_data = labels->data<int64_t>();
      T* logit_grad_data = logit_grad->data<T>();
      const T* out_grad_data = out_grad->data<T>();
      for (int i = 0; i < batch_size; ++i) {
        logit_grad_data[i * class_num + label_data[i]] -= out_grad_data[i];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
