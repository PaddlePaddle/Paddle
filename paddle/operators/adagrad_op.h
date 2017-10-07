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
class AdagradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out_tensor = ctx.Output<framework::Tensor>("ParamOut");
    auto moment_out_tensor = ctx.Output<framework::Tensor>("MomentOut");

    param_out_tensor->mutable_data<T>(ctx.GetPlace());
    moment_out_tensor->mutable_data<T>(ctx.GetPlace());

    float epsilon = ctx.Attr<float>("epsilon");

    auto param = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Param"));
    auto grad = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Grad"));
    auto moment = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("Moment"));
    auto lr = framework::EigenVector<T>::Flatten(
        *ctx.Input<framework::Tensor>("LearningRate"));

    auto param_out = framework::EigenVector<T>::Flatten(*param_out_tensor);
    auto moment_out = framework::EigenVector<T>::Flatten(*moment_out_tensor);
    auto place = ctx.GetEigenDevice<Place>();

    moment_out.device(place) = moment + grad * grad;
    Eigen::DSizes<int, 1> m_dsize(moment_out_tensor->numel());
    param_out.device(place) =
        param - lr.broadcast(m_dsize) * grad / (moment_out.sqrt() + epsilon);
  }
};

}  // namespace operators
}  // namespace paddle
