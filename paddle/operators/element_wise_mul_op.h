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
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class ElemWiseMulOPKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Y = context.Input<Tensor>("Y");
    auto* Z = context.Output<Tensor>("Out");
    Z->mutable_data<T>(context.GetPlace());
    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);

    auto X_e = EigenMatrix<T>::From(*X);
    auto Y_e = EigenMatrix<T>::From(*Y);
    auto Z_e = EigenMatrix<T>::From(*Z);

    // TODO: gpu?
    Z_e.device(context.GetEigenDevice<place>()) = X_e.cwiseProduct(Y_e);
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
    auto* device_context =
        const_cast<platform::DeviceContext*>(ctx.device_context_);

    auto X_e = EigenMatrix<T>::From(*X);
    auto Y_e = EigenMatrix<T>::From(*Y);
    auto dX_e = EigenMatrix<T>::From(*dX);
    auto dY_e = EigenMatrix<T>::From(*dY);
    auto dOut_e = EigenMatrix<T>::From(*dOut);

    // TODO: gpu?
    dX.device(context.GetEigenDevice<place>()) = dOut_e.cwiseProduct(Y_e);
    dY.device(context.GetEigenDevice<place>()) = dOut_e.cwiseProduct(X_e);
  }
};

}  // namespace operators
}
