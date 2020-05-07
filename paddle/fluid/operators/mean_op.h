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
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class MeanKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");

    output->mutable_data<T>(context.GetPlace());

    auto X = EigenVector<T>::Flatten(*input);
    auto y = EigenScalar<T>::From(*output);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    y.device(place) = X.mean();
  }
};

template <typename DeviceContext, typename T>
class MeanGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto OG = context.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(OG->numel(), 1UL,
                      platform::errors::InvalidArgument(
                          "Mean Gradient should be scalar. But received "
                          "Out@Grad's elements num is %d.",
                          OG->numel()));
    auto IG = context.Output<Tensor>(framework::GradVarName("X"));
    IG->mutable_data<T>(context.GetPlace());

    T ig_size = static_cast<T>(IG->numel());
    Eigen::DSizes<int, 1> bcast(static_cast<int>(ig_size));
    EigenVector<T>::Flatten(*IG).device(
        *context.template device_context<DeviceContext>().eigen_device()) =
        (EigenVector<T>::From(*OG) / ig_size).broadcast(bcast);
  }
};

}  // namespace operators
}  // namespace paddle
