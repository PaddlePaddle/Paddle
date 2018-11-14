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

template <typename DeviceContext, typename T>
class RankLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_t = ctx.Output<framework::Tensor>("Out");
    auto* label_t = ctx.Input<framework::Tensor>("Label");
    auto* left_t = ctx.Input<framework::Tensor>("Left");
    auto* right_t = ctx.Input<framework::Tensor>("Right");
    out_t->mutable_data<T>(ctx.GetPlace());

    auto out = framework::EigenVector<T>::Flatten(*out_t);
    auto label = framework::EigenVector<T>::Flatten(*label_t);
    auto left = framework::EigenVector<T>::Flatten(*left_t);
    auto right = framework::EigenVector<T>::Flatten(*right_t);

    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    out.device(dev) =
        (1. + (left - right).exp()).log() - label * (left - right);
  }
};

template <typename DeviceContext, typename T>
class RankLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_left_t =
        ctx.Output<framework::Tensor>(framework::GradVarName("Left"));
    auto* d_right_t =
        ctx.Output<framework::Tensor>(framework::GradVarName("Right"));

    auto* d_out_t = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* label_t = ctx.Input<framework::Tensor>("Label");
    auto* left_t = ctx.Input<framework::Tensor>("Left");
    auto* right_t = ctx.Input<framework::Tensor>("Right");

    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    auto d_out = framework::EigenVector<T>::Flatten(*d_out_t);
    auto label = framework::EigenVector<T>::Flatten(*label_t);
    auto left = framework::EigenVector<T>::Flatten(*left_t);
    auto right = framework::EigenVector<T>::Flatten(*right_t);

    // compute d_left
    if (d_left_t) {
      d_left_t->mutable_data<T>(ctx.GetPlace());
      auto d_left = framework::EigenVector<T>::Flatten(*d_left_t);
      d_left.device(dev) = d_out * (1. / (1. + (right - left).exp()) - label);
    }
    // compute d_right
    if (d_right_t) {
      d_right_t->mutable_data<T>(ctx.GetPlace());
      auto d_right = framework::EigenVector<T>::Flatten(*d_right_t);
      d_right.device(dev) =
          -d_out * (1.0 / (1. + (right - left).exp()) - label);
    }
  }
};
}  // namespace operators
}  // namespace paddle
