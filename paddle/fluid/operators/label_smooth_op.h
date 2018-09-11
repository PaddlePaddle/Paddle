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
class LabelSmoothKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_t = ctx.Output<framework::LoDTensor>("Out");
    auto* in_t = ctx.Input<framework::LoDTensor>("X");
    auto* dist_t = ctx.Input<framework::Tensor>("PriorDist");
    auto label_dim = in_t->dims()[1];
    out_t->mutable_data<T>(ctx.GetPlace());

    auto epsilon = ctx.Attr<float>("epsilon");
    auto out = framework::EigenVector<T>::Flatten(*out_t);
    auto in = framework::EigenVector<T>::Flatten(*in_t);
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    if (dist_t) {
      auto dist = framework::EigenVector<T>::Flatten(*dist_t);
      out.device(dev) =
          static_cast<T>(1 - epsilon) * in +
          static_cast<T>(epsilon) *
              dist.broadcast(Eigen::DSizes<int, 1>(in_t->numel()));
    } else {
      out.device(dev) = static_cast<T>(1 - epsilon) * in +
                        static_cast<T>(epsilon / label_dim);
    }
  }
};

template <typename DeviceContext, typename T>
class LabelSmoothGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_out_t = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_in_t = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    d_in_t->mutable_data<T>(ctx.GetPlace());

    auto d_out = framework::EigenVector<T>::Flatten(*d_out_t);
    auto d_in = framework::EigenVector<T>::Flatten(*d_in_t);

    auto epsilon = ctx.Attr<float>("epsilon");
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    d_in.device(dev) = static_cast<T>(1 - epsilon) * d_out;
  }
};
}  // namespace operators
}  // namespace paddle
