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
class MomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out = ctx.Output<Tensor>("ParamOut");
    auto velocity_out = ctx.Output<Tensor>("VelocityOut");

    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<T>(ctx.GetPlace());

    float mu = ctx.Attr<float>("mu");

    auto param = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Param"));
    auto grad = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Grad"));
    auto velocity = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Velocity"));
    float learning_rate = ctx.Input<Tensor>("LearningRate")->data<float>()[0];
    auto p_out = EigenVector<T>::Flatten(*param_out);
    auto v_out = EigenVector<T>::Flatten(*velocity_out);
    auto place = ctx.GetEigenDevice<Place>();

    v_out.device(place) = velocity * mu + grad;
    p_out.device(place) = param - learning_rate * v_out;
  }
};

}  // namespace operators
}  // namespace paddle
