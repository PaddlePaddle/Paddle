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

#include <cmath>

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename Place, typename T>
class AdamOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // TODO(tonyyang-svail): should be int, but here is float
    auto p = EigenVector<T>::Flatten(*ctx.Input<Tensor>("param"));
    auto g = EigenVector<T>::Flatten(*ctx.Input<Tensor>("grad"));
    auto m1 = EigenVector<T>::Flatten(*ctx.Input<Tensor>("moment1"));
    auto m2 = EigenVector<T>::Flatten(*ctx.Input<Tensor>("moment2"));

    ctx.Output<Tensor>("param_out")->mutable_data<T>(ctx.GetPlace());
    ctx.Output<Tensor>("moment1_out")->mutable_data<T>(ctx.GetPlace());
    ctx.Output<Tensor>("moment2_out")->mutable_data<T>(ctx.GetPlace());
    auto p_o = EigenVector<T>::Flatten(*ctx.Output<Tensor>("param_out"));
    auto m1_o = EigenVector<T>::Flatten(*ctx.Output<Tensor>("moment1_out"));
    auto m2_o = EigenVector<T>::Flatten(*ctx.Output<Tensor>("moment2_out"));

    int t = ctx.Attr<int>("time_step");
    float lr = ctx.Attr<float>("learning_rate");
    float beta1 = ctx.Attr<float>("beta1");
    float beta2 = ctx.Attr<float>("beta2");
    float epsilon = ctx.Attr<float>("epsilon");

    float beta1_to_t = std::pow(beta1, t);
    float beta2_to_t = std::pow(beta2, t);

    m1_o = beta1 * m1 + (1 - beta1) * g;
    m2_o = beta2 * m2 + (1 - beta2) * g;
    auto m1_hat = m1_o / (1 - beta1_to_t);
    auto m2_hat = m2_o / (1 - beta2_to_t);

    auto place = ctx.GetEigenDevice<Place>();
    p_o.device(place) = p - lr * m1_hat / (m2_hat.sqrt() + epsilon);
  }
};

}  // namespace operators
}  // namespace paddle
