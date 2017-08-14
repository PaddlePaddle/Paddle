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
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair = {
        {Eigen::IndexPair<Eigen::DenseIndex>(1, 0)}};
    auto* input0 = context.Input<Tensor>("X");
    auto* input1 = context.Input<Tensor>("Y");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());
    auto X = EigenMatrix<T>::From(*input0);
    auto Y = EigenMatrix<T>::From(*input1);
    auto Z = EigenMatrix<T>::From(*output);
    auto& place = context.GetEigenDevice<Place>();

    Z.device(place) = X.contract(Y, dim_pair);
  }
};

template <typename Place, typename T>
class MulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input0 = ctx.Input<Tensor>("X");
    auto* input1 = ctx.Input<Tensor>("Y");
    auto* input2 = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* output0 = ctx.Output<Tensor>(0);
    auto* output1 = ctx.Output<Tensor>(1);
    output0->mutable_data<T>(ctx.GetPlace());
    output1->mutable_data<T>(ctx.GetPlace());

    auto X = EigenMatrix<T>::From(*input0);
    auto Y = EigenMatrix<T>::From(*input1);
    auto dOut = EigenMatrix<T>::From(*input2);
    auto dX = EigenMatrix<T>::From(*output0);
    auto dY = EigenMatrix<T>::From(*output1);

    // dX = Out@G * Y'
    // dY = X' * Out@G
    auto place = ctx.GetEigenDevice<Place>();
    // TODO(dzh,qijun) : need transpose feature of blas library
    // Eigen Tensor does not support it very well
    // dX.device(place) = dOut.contract(dOut, transpose)
  }
};

}  // namespace operators
}  // namespace paddle
