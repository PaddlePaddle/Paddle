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

namespace paddle {
namespace operators {

template <typename Place, typename T>
class AdamOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out_tensor = ctx.Output<framework::Tensor>("ParamOut");
    auto moment1_out_tensor = ctx.Output<framework::Tensor>("Moment1Out");
    auto moment2_out_tensor = ctx.Output<framework::Tensor>("Moment2Out");

    param_out_tensor->mutable_data<T>(ctx.GetPlace());
    moment1_out_tensor->mutable_data<T>(ctx.GetPlace());
    moment2_out_tensor->mutable_data<T>(ctx.GetPlace());

    float beta1 = ctx.Attr<float>("beta1");
    float beta2 = ctx.Attr<float>("beta2");
    float epsilon = ctx.Attr<float>("epsilon");

    auto param = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Param"));
    auto grad = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Grad"));
    auto moment1 = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Moment1"));
    auto moment2 = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Moment2"));
    auto lr = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("LearningRate"));
    auto beta1_pow = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Beta1Pow"));
    auto beta2_pow = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Beta2Pow"));
    auto param_out = framework::EigenVector<T>::Flatten(*param_out_tensor);
    auto moment1_out = framework::EigenVector<T>::Flatten(*moment1_out_tensor);
    auto moment2_out = framework::EigenVector<T>::Flatten(*moment2_out_tensor);
    auto place = ctx.GetEigenDevice<Place>();

    moment1_out.device(place) = beta1 * moment1 + (1 - beta1) * grad;
    moment2_out.device(place) = beta2 * moment2 + (1 - beta2) * grad.square();

    // All of these are tensors of 1 element
    auto lr_t = lr * (1 - beta2_pow).sqrt() / (1 - beta1_pow);
    // Eigen does not support automatic broadcast
    // Get dimensions of moment vector to broadcast lr_t
    Eigen::DSizes<int, 1> m_dsize(moment1_out_tensor->numel());
    param_out.device(place) =
        param -
        lr_t.broadcast(m_dsize) *
            (moment1_out / (moment2_out.sqrt() + epsilon));
  }
};

}  // namespace operators
}  // namespace paddle
