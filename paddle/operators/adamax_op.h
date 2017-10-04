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
class AdamaxOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out = ctx.Output<Tensor>("param_out");
    auto moment_out = ctx.Output<Tensor>("moment_out");
    auto norm_out = ctx.Output<Tensor>("inf_norm_out");

    param_out->mutable_data<T>(ctx.GetPlace());
    moment_out->mutable_data<T>(ctx.GetPlace());
    norm_out->mutable_data<T>(ctx.GetPlace());

    float beta_1 = ctx.Attr<float>("beta_1");
    float beta_2 = ctx.Attr<float>("beta_2");
    float epsilon = ctx.Attr<float>("epsilon");
    auto lr = ctx.Input<Tensor>("learning_rate")->data<float>()[0];
    auto t = ctx.Input<Tensor>("time_step")->data<int>()[0];

    auto p = EigenVector<T>::Flatten(*ctx.Input<Tensor>("param"));
    auto g = EigenVector<T>::Flatten(*ctx.Input<Tensor>("grad"));
    auto m = EigenVector<T>::Flatten(*ctx.Input<Tensor>("moment"));
    auto u = EigenVector<T>::Flatten(*ctx.Input<Tensor>("inf_norm"));
    auto p_out = EigenVector<T>::Flatten(*param_out);
    auto m_out = EigenVector<T>::Flatten(*moment_out);
    auto u_out = EigenVector<T>::Flatten(*norm_out);
    auto place = ctx.GetEigenDevice<Place>();

    m_out.device(place) = beta_1 * m + (1 - beta_1) * g;
    u_out.device(place) = g.abs().cwiseMax((beta_2 * u) + epsilon);

    float lr_t = lr / (1 - std::pow(beta_1, t));
    p_out.device(place) = p - lr_t * (m_out / u_out);
  }
};

}  // namespace operators
}  // namespace paddle
