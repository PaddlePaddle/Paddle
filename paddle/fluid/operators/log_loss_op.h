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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T, typename AttrType = T>
class LogLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* loss_out = ctx.Output<Tensor>("Loss");

    loss_out->mutable_data<T>(ctx.GetPlace());

    auto epsilon = static_cast<T>(ctx.Attr<AttrType>("epsilon"));

    auto prediction = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Predicted"));
    auto label = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Labels"));

    auto loss = EigenVector<T>::Flatten(*loss_out);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    loss.device(place) = (-(label * (prediction + epsilon).log()) -
                          ((static_cast<T>(1) - label) *
                           (static_cast<T>(1) - prediction + epsilon).log()));
  }
};

template <typename DeviceContext, typename T, typename AttrType = T>
class LogLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto epsilon = static_cast<T>(ctx.Attr<AttrType>("epsilon"));

    auto prediction = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Predicted"));
    auto label = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Labels"));

    auto* dloss = ctx.Input<Tensor>(framework::GradVarName("Loss"));
    auto* dpred = ctx.Output<Tensor>(framework::GradVarName("Predicted"));

    auto dl = EigenVector<T>::Flatten(*dloss);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    if (dpred) {
      dpred->mutable_data<T>(ctx.GetPlace());
      auto dx = framework::EigenVector<T>::Flatten(*dpred);
      dx.device(place) = dl * (-(label / (prediction + epsilon)) +
                               ((static_cast<T>(1) - label) /
                                (static_cast<T>(1) - prediction + epsilon)));
    }
  }
};

}  // namespace operators
}  // namespace paddle
