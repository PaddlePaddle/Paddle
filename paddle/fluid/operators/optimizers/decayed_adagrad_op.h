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
class DecayedAdagradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(),
                      true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));
    const auto* grad_var = ctx.InputVar("Grad");
    PADDLE_ENFORCE_EQ(grad_var->IsType<framework::LoDTensor>(),
                      true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Grad").front(),
                          framework::ToTypeName(grad_var->Type())));

    auto param_out_tensor = ctx.Output<phi::DenseTensor>("ParamOut");
    auto moment_out_tensor = ctx.Output<phi::DenseTensor>("MomentOut");

    param_out_tensor->mutable_data<T>(ctx.GetPlace());
    moment_out_tensor->mutable_data<T>(ctx.GetPlace());

    float decay = ctx.Attr<float>("decay");
    float epsilon = ctx.Attr<float>("epsilon");

    auto param = framework::EigenVector<T>::Flatten(
        *ctx.Input<phi::DenseTensor>("Param"));
    auto grad = framework::EigenVector<T>::Flatten(
        *ctx.Input<phi::DenseTensor>("Grad"));
    auto moment = framework::EigenVector<T>::Flatten(
        *ctx.Input<phi::DenseTensor>("Moment"));
    auto lr = framework::EigenVector<T>::Flatten(
        *ctx.Input<phi::DenseTensor>("LearningRate"));

    auto param_out = framework::EigenVector<T>::Flatten(*param_out_tensor);
    auto moment_out = framework::EigenVector<T>::Flatten(*moment_out_tensor);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    moment_out.device(place) = decay * moment + (1 - decay) * grad * grad;
    Eigen::DSizes<int, 1> m_dsize(moment_out_tensor->numel());
    param_out.device(place) =
        param - lr.broadcast(m_dsize) * grad / (moment_out.sqrt() + epsilon);
  }
};

}  // namespace operators
}  // namespace paddle
