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
class AdadeltaOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE(param_var->IsType<framework::LoDTensor>(),
                   "The Var(%s)'s type should be LoDTensor, "
                   "but the received is %s",
                   ctx.Inputs("Param").front(),
                   framework::ToTypeName(param_var->Type()));
    const auto* grad_var = ctx.InputVar("Grad");
    PADDLE_ENFORCE(grad_var->IsType<framework::LoDTensor>(),
                   "The Var(%s)'s type should be LoDTensor, "
                   "but the received is %s",
                   ctx.Inputs("Grad").front(),
                   framework::ToTypeName(grad_var->Type()));

    auto param_out_tensor = ctx.Output<framework::Tensor>("ParamOut");
    auto avg_squared_grad_out_tensor =
        ctx.Output<framework::Tensor>("AvgSquaredGradOut");
    auto avg_squared_update_out_tensor =
        ctx.Output<framework::Tensor>("AvgSquaredUpdateOut");

    param_out_tensor->mutable_data<T>(ctx.GetPlace());
    avg_squared_grad_out_tensor->mutable_data<T>(ctx.GetPlace());
    avg_squared_update_out_tensor->mutable_data<T>(ctx.GetPlace());

    T rho = static_cast<T>(ctx.Attr<float>("rho"));
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));

    auto param = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Param"));
    auto grad = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Grad"));
    // Squared gradient accumulator
    auto avg_squared_grad = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("AvgSquaredGrad"));
    // Squared updates accumulator
    auto avg_squared_update = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("AvgSquaredUpdate"));
    auto param_out = framework::EigenVector<T>::Flatten(*param_out_tensor);
    auto avg_squared_grad_out =
        framework::EigenVector<T>::Flatten(*avg_squared_grad_out_tensor);
    auto avg_squared_update_out =
        framework::EigenVector<T>::Flatten(*avg_squared_update_out_tensor);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    avg_squared_grad_out.device(place) =
        rho * avg_squared_grad + (1 - rho) * grad.square();
    auto update =
        -((avg_squared_update + epsilon) / (avg_squared_grad_out + epsilon))
             .sqrt() *
        grad;
    avg_squared_update_out.device(place) =
        rho * avg_squared_update + (1 - rho) * update.square();
    param_out.device(place) = param + update;
  }
};

}  // namespace operators
}  // namespace paddle
