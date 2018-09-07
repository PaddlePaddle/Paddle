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

template <typename DeviceContext, typename T>
class RmspropOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* param_out = ctx.Output<Tensor>("ParamOut");
    auto* moment_out = ctx.Output<Tensor>("MomentOut");
    auto* mean_square_out = ctx.Output<Tensor>("MeanSquareOut");

    auto grad = ctx.Input<Tensor>("Grad");

    param_out->mutable_data<T>(ctx.GetPlace());
    moment_out->mutable_data<T>(ctx.GetPlace());
    mean_square_out->mutable_data<T>(ctx.GetPlace());

    float epsilon = ctx.Attr<float>("epsilon");
    float rho = ctx.Attr<float>("decay");
    float momentum = ctx.Attr<float>("momentum");

    auto p = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Param"));
    auto ms = EigenVector<T>::Flatten(*ctx.Input<Tensor>("MeanSquare"));
    auto lr = EigenVector<T>::Flatten(*ctx.Input<Tensor>("LearningRate"));
    auto g = EigenVector<T>::Flatten(*grad);
    auto mom = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Moment"));

    auto p_out = EigenVector<T>::Flatten(*param_out);
    auto mom_out = EigenVector<T>::Flatten(*moment_out);
    auto ms_out = EigenVector<T>::Flatten(*mean_square_out);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    Eigen::DSizes<int, 1> grad_dsize(grad->numel());

    ms_out.device(place) = rho * ms + (1 - rho) * g * g;
    mom_out.device(place) =
        momentum * mom +
        lr.broadcast(grad_dsize) * g / (ms_out + epsilon).sqrt();
    p_out.device(place) = p - mom_out;
  }
};

}  // namespace operators
}  // namespace paddle
