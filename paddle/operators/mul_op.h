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

#include "paddle/operators/math/math_function.h"

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class MulKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair = {
    //     {Eigen::IndexPair<Eigen::DenseIndex>(1, 0)}};
    auto* X = context.Input<Tensor>("X");
    auto* Y = context.Input<Tensor>("Y");
    auto* Z = context.Output<Tensor>("Out");
    Z->mutable_data<T>(context.GetPlace());
    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);
    math::matmul<Place, T>(*X, false, *Y, false, 1, Z, 0, device_context);

    // auto X = EigenMatrix<T>::From(*input0);
    // auto Y = EigenMatrix<T>::From(*input1);
    // auto Z = EigenMatrix<T>::From(*output);
    // auto& place = context.GetEigenDevice<Place>();

    // Z.device(place) = X.contract(Y, dim_pair);
  }
};

template <typename Place, typename T>
class MulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>("X");
    auto* Y = ctx.Input<Tensor>("Y");
    auto* dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dY = ctx.Output<Tensor>(framework::GradVarName("Y"));
    // auto* dXdata = dX->template mutable_data<T>(ctx.GetPlace());
    // auto* dYdata = dY->template mutable_data<T>(ctx.GetPlace());
    auto* device_context =
        const_cast<platform::DeviceContext*>(ctx.device_context_);
    math::matmul<Place, T>(*dOut, false, *Y, true, 1, dX, 0, device_context);
    math::matmul<Place, T>(*X, true, *dOut, false, 1, dY, 0, device_context);

    // auto X = EigenMatrix<T>::From(*input0);
    // auto Y = EigenMatrix<T>::From(*input1);
    // auto dOut = EigenMatrix<T>::From(*input2);
    // auto dX = EigenMatrix<T>::From(*output0);
    // auto dY = EigenMatrix<T>::From(*output1);

    // dX = Out@G * Y'
    // dY = X' * Out@G
    // auto place = ctx.GetEigenDevice<Place>();
    // TODO(dzh,qijun) : need transpose feature of blas library
    // Eigen Tensor does not support it very well
    // dX.device(place) = matmul(input2, )
  }
};

}  // namespace operators
}  // namespace paddle
