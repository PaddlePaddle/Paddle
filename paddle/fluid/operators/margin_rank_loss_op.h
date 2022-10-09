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

template <typename T>
struct ReLU {
  HOSTDEVICE T operator()(const T& val) const {
    return val > 0 ? val : static_cast<T>(0);
  }
};

template <typename T>
struct Heaviside {
  HOSTDEVICE T operator()(const T& val) const {
    return static_cast<T>(val > 0 ? 1 : 0);
  }
};

template <typename DeviceContext, typename T>
class MarginRankLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_t = ctx.Output<phi::DenseTensor>("Out");
    auto* act_t = ctx.Output<phi::DenseTensor>("Activated");

    auto* label_t = ctx.Input<phi::DenseTensor>("Label");
    auto* x1_t = ctx.Input<phi::DenseTensor>("X1");
    auto* x2_t = ctx.Input<phi::DenseTensor>("X2");

    out_t->mutable_data<T>(ctx.GetPlace());
    act_t->mutable_data<T>(ctx.GetPlace());

    auto margin = static_cast<T>(ctx.Attr<T>("margin"));
    auto out = framework::EigenVector<T>::Flatten(*out_t);
    auto act = framework::EigenVector<T>::Flatten(*act_t);

    auto label = framework::EigenVector<T>::Flatten(*label_t);
    auto x1 = framework::EigenVector<T>::Flatten(*x1_t);
    auto x2 = framework::EigenVector<T>::Flatten(*x2_t);

    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    out.device(dev) = (-label * (x1 - x2) + margin).unaryExpr(ReLU<T>());
    act.device(dev) = out.unaryExpr(Heaviside<T>());
  }
};

template <typename DeviceContext, typename T>
class MarginRankLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_x1_t =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X1"));
    auto* d_x2_t =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X2"));

    auto* act_t = ctx.Input<phi::DenseTensor>("Activated");
    auto* d_out_t = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* label_t = ctx.Input<phi::DenseTensor>("Label");

    auto d_out = framework::EigenVector<T>::Flatten(*d_out_t);
    auto act = framework::EigenVector<T>::Flatten(*act_t);
    auto label = framework::EigenVector<T>::Flatten(*label_t);
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();

    // compute d_x1
    if (d_x1_t) {
      d_x1_t->mutable_data<T>(ctx.GetPlace());
      auto d_x1 = framework::EigenVector<T>::Flatten(*d_x1_t);
      d_x1.device(dev) = -d_out * act * label;
    }
    // compute d_x2
    if (d_x2_t) {
      d_x2_t->mutable_data<T>(ctx.GetPlace());
      auto d_x2 = framework::EigenVector<T>::Flatten(*d_x2_t);
      d_x2.device(dev) = d_out * act * label;
    }
  }
};
}  // namespace operators
}  // namespace paddle
