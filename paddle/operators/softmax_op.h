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
#include "paddle/operators/math/softmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class SoftmaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto X = context.Input<Tensor>("X");
    auto Y = context.Output<Tensor>("Y");

    // allocate memory on device.
    Y->mutable_data<T>(context.GetPlace());

    math::SoftmaxFunctor<Place, T>()(context.device_context(), X, Y);
  }
};

template <typename Place, typename T>
class SoftmaxGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto Y = context.Input<Tensor>("Y");
    auto dY = context.Input<Tensor>(framework::GradVarName("Y"));
    auto dX = context.Output<Tensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(context.GetPlace());

    const int batch_size = Y->dims()[0];
    const int class_num = Y->dims()[1];

    Eigen::DSizes<int, 1> along_class(1);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, class_num);

    auto Y_eigen = EigenMatrix<T>::From(*Y);
    auto dY_eigen = EigenMatrix<T>::From(*dY);
    auto dX_eigen = EigenMatrix<T>::From(*dX);
    auto place = context.GetEigenDevice<Place>();

    auto dot = (Y_eigen * dY_eigen)
                   .sum(along_class)
                   .eval()
                   .reshape(batch_by_one)
                   .broadcast(one_by_class);
    dX_eigen.device(place) = (dY_eigen - dot) * Y_eigen;
  }
};

}  // namespace operators
}  // namespace paddle
