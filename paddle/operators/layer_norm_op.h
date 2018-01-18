/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

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

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LayerNormOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto out_tensor = ctx.Output<framework::Tensor>("Out");
    out_tensor->mutable_data<T>(ctx.GetPlace());
    auto out = framework::EigenVector<T>::Flatten(*out_tensor);
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));

    auto* input_tensor = ctx.Input<framework::Tensor>("X");
    auto batch_size = input_tensor->dims()[0];
    auto input_dim = input_tensor->dims()[1];

    auto input = framework::EigenVector<T>::Flatten(*input_tensor);
    auto scale = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Scale"));
    auto bias = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Bias"));

    auto* place = ctx.template device_context<DeviceContext>().eigen_device();

    auto mean = input.rowwise().mean();
    auto inv_std =
        ((((input.square()).rowwise()).mean() - mean.square()) + epsilon)
            .sqrt()
            .inverse();
    out.device(place) = scale.transpose().replicate(batch_size, 1) *
                            (input - mean.replicate(1, input_dim)) *
                            inv_std.replicate(1, input_dim) +
                        bias.transpose().replicate(batch_size, 1);
  }
};

template <typename T>
class LayerNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
