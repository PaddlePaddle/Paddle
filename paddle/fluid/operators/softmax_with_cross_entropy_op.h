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
#include "paddle/fluid/operators/softmax_op.h"

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
    const bool soft_label = context.Attr<bool>("soft_label");

    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logits->dims()[axis];

    softmax->mutable_data<T>(context.GetPlace());
    loss->mutable_data<T>(context.GetPlace());

    const int n = SizeToAxis(axis, logits->dims());
    const int d = SizeFromAxis(axis, logits->dims());
    Tensor logits_2d, softmax_2d, labels_2d, loss_2d;
    logits_2d.ShareDataWith(*logits).Resize({n, d});
    softmax_2d.ShareDataWith(*softmax).Resize({n, d});
    labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
    loss_2d.ShareDataWith(*loss).Resize({n, d / axis_dim});

    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    math::SoftmaxFunctor<platform::CPUDeviceContext, T, false>()(
        dev_ctx, axis_dim, &logits_2d, &softmax_2d);
    math::CrossEntropyFunctor<platform::CPUDeviceContext, T>()(
        dev_ctx, &loss_2d, &softmax_2d, &labels_2d, soft_label,
        context.Attr<int>("ignore_index"), axis_dim);
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

    const Tensor* softmax = context.Input<Tensor>("Softmax");
    if (logit_grad != softmax) {
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), logit_grad);
    }

    const bool soft_label = context.Attr<bool>("soft_label");

    const int rank = logit_grad->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logit_grad->dims()[axis];

    const int n = SizeToAxis(axis, logit_grad->dims());
    const int d = SizeFromAxis(axis, logit_grad->dims());
    Tensor logit_grad_2d, labels_2d, out_grad_2d;
    logit_grad_2d.ShareDataWith(*logit_grad).Resize({n, d});
    labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
    out_grad_2d.ShareDataWith(*out_grad).Resize({n, d / axis_dim});

    auto out_grad_mat = EigenMatrix<T>::From(out_grad_2d);
    auto logit_grad_mat = EigenMatrix<T>::From(logit_grad_2d);
    auto& place = *context.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    if (soft_label) {
      auto lbl_mat = EigenMatrix<T>::From(labels_2d);
      logit_grad_mat.device(place) =
          out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, axis_dim)) *
          (logit_grad_mat - lbl_mat);
    } else {
      logit_grad_mat.device(place) =
          logit_grad_mat *
          out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, axis_dim));

      const int64_t* label_data = labels->data<int64_t>();
      T* logit_grad_data = logit_grad->data<T>();
      const T* out_grad_data = out_grad->data<T>();
      const int remain = d / axis_dim;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < remain; j++) {
          int idx = i * remain + j;
          logit_grad_data[i * d + label_data[idx] * remain + j] -=
              out_grad_data[idx];
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
