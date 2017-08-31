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
class ElemWiseMulKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Y = context.Input<Tensor>("Y");
    auto* Z = context.Output<Tensor>("Out");
    Z->mutable_data<T>(context.GetPlace());

    auto X_e = framework::EigenVector<T>::Flatten(*X);
    auto Y_e = framework::EigenVector<T>::Flatten(*Y);
    auto Z_e = framework::EigenVector<T>::Flatten(*Z);

    Z_e.device(context.GetEigenDevice<Place>()) = X_e * Y_e;
  }
};

template <typename Place, typename T>
class ElemWiseMulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>("X");
    auto* Y = ctx.Input<Tensor>("Y");
    auto* dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dY = ctx.Output<Tensor>(framework::GradVarName("Y"));
    dX->mutable_data<T>(ctx.GetPlace());
    dY->mutable_data<T>(ctx.GetPlace());

    auto X_e = framework::EigenVector<T>::Flatten(*X);
    auto Y_e = framework::EigenVector<T>::Flatten(*Y);
    auto dX_e = framework::EigenVector<T>::Flatten(*dX);
    auto dY_e = framework::EigenVector<T>::Flatten(*dY);
    auto dOut_e = framework::EigenVector<T>::Flatten(*dOut);

    dX_e.device(ctx.GetEigenDevice<Place>()) = dOut_e * Y_e;
    dY_e.device(ctx.GetEigenDevice<Place>()) = X_e * dOut_e;
  }
};

}  // namespace operators
}  // namespace paddle
