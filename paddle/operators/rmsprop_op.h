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
class RmspropOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out = ctx.Output<Tensor>("ParamOut");
    auto moment_out = ctx.Output<Tensor>("MomentOut");

    param_out->mutable_data<T>(ctx.GetPlace());
    moment_out->mutable_data<T>(ctx.GetPlace());

    float epsilon = ctx.Attr<float>("epsilon");
    float decay = ctx.Attr<float>("decayRate");

    auto p = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Param"));
    auto g = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Grad"));
    auto m = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Moment"));
    float lr = ctx.Input<Tensor>("LearningRate")->data<float>()[0];
    auto p_out = EigenVector<T>::Flatten(*param_out);
    auto m_out = EigenVector<T>::Flatten(*moment_out);
    auto place = ctx.GetEigenDevice<Place>();

    m_out.device(place) = decay * m + (1 - decay) * g * g;
    p_out.device(place) = p - lr * g / (m_out.sqrt() + epsilon);
  }
};

}  // namespace operators
}  // namespace paddle
