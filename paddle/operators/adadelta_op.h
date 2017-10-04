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

using Tensor = framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename Place, typename T>
class AdadeltaOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out = ctx.Output<Tensor>("ParamOut");
    auto avg_squared_grad_out = ctx.Output<Tensor>("AvgSquaredGradOut");
    auto avg_squared_update_out = ctx.Output<Tensor>("AvgSquaredUpdateOut");

    param_out->mutable_data<T>(ctx.GetPlace());
    avg_squared_grad_out->mutable_data<T>(ctx.GetPlace());
    avg_squared_update_out->mutable_data<T>(ctx.GetPlace());

    float rho = ctx.Attr<float>("rho");
    float epsilon = ctx.Attr<float>("epsilon");

    auto p = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Param"));
    auto g = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Grad"));
    // Squared gradient accumulator
    auto g_acc = EigenVector<T>::Flatten(*ctx.Input<Tensor>("AvgSquaredGrad"));
    // Squared updates accumulator
    auto u_acc =
        EigenVector<T>::Flatten(*ctx.Input<Tensor>("AvgSquaredUpdate"));
    auto p_out = EigenVector<T>::Flatten(*param_out);
    auto g_acc_out = EigenVector<T>::Flatten(*avg_squared_grad_out);
    auto u_acc_out = EigenVector<T>::Flatten(*avg_squared_update_out);
    auto place = ctx.GetEigenDevice<Place>();

    g_acc_out.device(place) = rho * g_acc + (1 - rho) * g.square();
    auto update = -((u_acc + epsilon) / (g_acc_out + epsilon)).sqrt() * g;
    u_acc_out.device(place) = rho * u_acc + (1 - rho) * update.square();
    p_out.device(place) = p + update;
  }
};

}  // namespace operators
}  // namespace paddle
