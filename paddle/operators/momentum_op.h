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
class MomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto velocity = ctx.Input<framework::Tensor>("Velocity");
    auto grad = ctx.Input<framework::Tensor>("Grad");
    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<T>(ctx.GetPlace());

    float mu = ctx.Attr<float>("mu");
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    auto p_out = framework::EigenVector<T>::Flatten(*param_out);
    auto v_out = framework::EigenVector<T>::Flatten(*velocity_out);

    auto p = framework::EigenVector<T>::Flatten(*param);
    auto v = framework::EigenVector<T>::Flatten(*velocity);
    auto g = framework::EigenVector<T>::Flatten(*grad);
    auto lr = framework::EigenVector<T>::Flatten(*learning_rate);

    auto place = ctx.GetEigenDevice<Place>();

    Eigen::DSizes<int, 1> grad_dsize(grad->numel());

    v_out.device(place) = v * mu + g;
    if (use_nesterov) {
      p_out.device(place) = p - g * lr.broadcast(grad_dsize) +
                            v_out * mu * lr.broadcast(grad_dsize);
    } else {
      p_out.device(place) = p - lr.broadcast(grad_dsize) * v_out;
    }
  }
};

}  // namespace operators
}  // namespace paddle
